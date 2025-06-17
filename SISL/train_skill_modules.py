import torch
import torch.optim as optim
import argparse
from typing import List
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb
import wandb
from tqdm import tqdm
import os
import time
import yaml

from models.skill_vae import SkillVAE
from data.skill_dataloader import SkillsDataset
from models.rnvp import stacked_NVP
from utils.general_utils import AttrDict

from undirected_si import *
from directed_si import *
import sys


class ModelTrainer():
    '''
        底层skill训练类
    '''
    def __init__(self, dataset_name, config_file, update_interval=500):
        '''
            构造函数，初始化模型训练器
        '''
        # 类属性初始化：数据集名称、结果及模型保存目录、配置文件路径、运行设备
        self.dataset_name = dataset_name
        self.save_dir = "./results/saved_skill_models/" + dataset_name +"/"
        os.makedirs(self.save_dir, exist_ok=True)
        self.vae_save_path = self.save_dir + "skill_vae.pth"
        self.sp_save_path = self.save_dir + "skill_prior.pth"
        config_path = "configs/skill_mdl/" + config_file

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)

        # 配置文件中加载模型及训练配置
        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)
            conf = AttrDict(conf)
        for key in conf:
            conf[key] = AttrDict(conf[key])        

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])          
        train_data = SkillsDataset(dataset_name, phase="train", subseq_len=conf.skill_vae.subseq_len, transform=transform)
        val_data   = SkillsDataset(dataset_name, phase="val", subseq_len=conf.skill_vae.subseq_len, transform=transform)

        # 创建数据加载器 Dataloader，包括训练集与测试集加载器
        self.train_loader = DataLoader(
            train_data,
            batch_size = conf.skill_vae.batch_size,
            shuffle = True,
            drop_last=True,
            prefetch_factor=30,
            num_workers=8,
            pin_memory=True)

        self.val_loader = DataLoader(
            val_data,
            batch_size = 64,
            shuffle = False,
            drop_last=True,
            prefetch_factor=30,
            num_workers=8,
            pin_memory=True)

        # 模型初始化：SkillVAE 与 StackedNVP
        self.update_interval = update_interval  # 每500个batch更新一次goal_state
        self.current_goal_state = None  # 用于缓存当前的goal_state
        self.current_obs_shape = None  # 用于检查obs形状是否变化
        
        self.skill_vae = SkillVAE(n_actions=conf.skill_vae.n_actions, 
                                 n_obs=conf.skill_vae.n_obs * 2,
                                 n_hidden=conf.skill_vae.n_hidden,
                                 seq_length=conf.skill_vae.subseq_len, 
                                 n_z=conf.skill_vae.n_z, 
                                 device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.skill_vae.parameters(), lr=conf.skill_vae.lr)

        self.sp_nvp = stacked_NVP(d=conf.skill_prior_nvp.d, k=conf.skill_prior_nvp.k, n_hidden=conf.skill_prior_nvp.n_hidden,
                                  state_size=conf.skill_vae.n_obs, n=conf.skill_prior_nvp.n_coupling_layers, device=self.device).to(self.device)
        
        # 构建优化器以及学习率调度器
        self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.sp_optimizer, 0.999)

        self.n_epochs = conf.skill_vae.epochs


    def fit(self, epoch):
        '''
            训练批数据上模型训练方法
        '''
        self.skill_vae.train()
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            # 训练数据加载
            data["actions"] = data["actions"].to(self.device)
            data["obs"] = data["obs"].to(self.device)

            # 检查是否需要更新goal_state
            need_update = (
                i % self.update_interval == 0 or  # 达到更新间隔
                self.current_goal_state is None or  # 第一次运行
                data["obs"].shape != self.current_obs_shape  # 输入形状变化
            )
            
            if need_update:
                # 计算新的goal_state
                pt, obs_dict = si_abstract(data["obs"])
                directed_si = Directed_Structural_Information(data["obs"], pt, obs_dict)
                self.current_goal_state = directed_si.get_community_representations()
                self.current_obs_shape = data["obs"].shape
                
                # 验证形状一致性
                assert self.current_obs_shape == self.current_goal_state.shape, \
                    f"Shape mismatch: obs {self.current_obs_shape} vs goal_state {self.current_goal_state.shape}"
            
            # 使用当前(可能是缓存的)goal_state
            goal_state = self.current_goal_state
            
            # 如果形状不匹配(如由于最后一个batch)，调整goal_state
            if goal_state.shape[0] != data["obs"].shape[0]:
                # 简单复制第一个元素来匹配batch大小
                goal_state = goal_state[0:1].expand(data["obs"].shape[0], *goal_state.shape[1:])
            
            # 拼接原始obs和goal_state作为新输入
            combined_obs = torch.cat([data["obs"], goal_state], dim=-1)
            
            # 修改后的SkillVAE训练
            self.skill_vae.init_hidden(data["actions"].size(0))
            self.optimizer.zero_grad()
            output = self.skill_vae(AttrDict(actions=data["actions"], obs=combined_obs))
            losses = self.skill_vae.loss(data, output)
            loss = losses.total_loss
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # StackedNVP 训练
            self.sp_optimizer.zero_grad()
            sp_input = AttrDict(skill=output.z.detach(),
                                state=data["obs"][:,0,:])
            z, log_pz, log_jacob = self.sp_nvp(sp_input)
            sp_loss = (-log_pz - log_jacob).mean()
            sp_loss.backward()
            self.sp_optimizer.step()

            # 调用wandb记录学习率以及训练损失
            if i % 500 == 0:
                self.scheduler.step()
                wandb.log({'lr':self.scheduler.get_lr()[0]}, epoch)

            if i % 100:
                wandb.log({'BC Loss_VAE':losses.bc_loss.item()}, epoch)
                wandb.log({'KL_Loss_VAE':losses.kld_loss.item()}, epoch)
                wandb.log({'NVP_Loss':sp_loss.item()}, epoch)
            
        train_loss = running_loss/len(self.train_loader.dataset)
        return train_loss


    def validate(self):
        '''
            整个验证集上模型评估方法
        '''
        self.skill_vae.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                data["actions"] = data["actions"].to(self.device)
                data["obs"] = data["obs"].to(self.device)
                
                # 验证集每次都计算新的goal_state以确保准确性
                pt, obs_dict = si_abstract(data["obs"])
                directed_si = Directed_Structural_Information(data["obs"], pt, obs_dict)
                goal_state = directed_si.get_community_representations()
                
                # 拼接输入
                combined_obs = torch.cat([data["obs"], goal_state], dim=-1)
                
                self.skill_vae.init_hidden(data["actions"].size(0))
                output = self.skill_vae(AttrDict(actions=data["actions"], obs=combined_obs))
                losses = self.skill_vae.loss(data, output)

                loss = losses.bc_loss.item()
                running_loss += loss

        val_loss = running_loss/len(self.val_loader.dataset)
        return val_loss


    def train(self):
        '''
            模型训练方法
        '''
        print("Training...")
        # 从训练集中，逐步加载批数据 
        for epoch in tqdm(range(self.n_epochs)):
            # 批数据训练，记录训练损失
            train_epoch_loss = self.fit(epoch)
            # 模型评估，记录测试损失
            if epoch%5 == 0:
                val_epoch_loss = self.validate()

            # 调用wandb记录训练与测试损失
            wandb.log({'train_loss':train_epoch_loss}, epoch)
            wandb.log({'val_loss':val_epoch_loss}, epoch)

            # 模型保存，包括 SkillVAE 与 StackedNVP
            if epoch % 50 == 0:
                torch.save(self.skill_vae, self.vae_save_path)
                torch.save(self.sp_nvp, self.sp_save_path)
                
   
if __name__ == "__main__":

    # 创建解析器，解析配置文件以及数据集名称
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="block/config.yaml")
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    args=parser.parse_args()
    
    # 初始化wandb，以追踪实验
    wandb.login(key="c12cabda70e7328a5f1a3182508b5ac39333274e")
    wandb.init(project="skill_mdl")  # ****配置用户key****
    wandb.run.name = "skill_mdl_" + time.asctime()
    wandb.run.save()

    # 构建ModelTrainer实例，进行训练
    trainer = ModelTrainer(args.dataset_name, args.config_file)
    trainer.train()
