---
layout: post
title: "Building GitHub Pages with RVM, Jekyll, and GitHub Actions Workflow"
date: 2023-05-07 17:34:56 +0800
categories: jekyll github-pages rvm github-actions
---
In this post, we will go through the process of building a GitHub Pages site using RVM, Jekyll, and a GitHub Actions Workflow. We'll cover key concepts, step-by-step instructions, and address the bugs encountered during the process.

## Key Concepts

- RVM (Ruby Version Manager): A command-line tool for installing, managing, and working with multiple Ruby environments.
Jekyll: A static site generator built with Ruby, used for creating GitHub Pages sites.
- GitHub Actions: A CI/CD platform integrated with GitHub, allowing you to automate workflows for building, testing, and deploying projects.

## Step-by-Step Guide

1. Install RVM: Follow the installation instructions for RVM on the official RVM website.
2. Install Ruby: Use RVM to install the desired version of Ruby. For example, to install Ruby 2.7.4, run rvm install 2.7.4 in your terminal.
3. Set up Jekyll: Install Jekyll by running gem install jekyll bundler.
Create a new Jekyll site: Run jekyll new my-site to create a new Jekyll site in the my-site directory. Replace my-site with your desired directory name.
4. Build and serve the site locally: Navigate to the site directory and run bundle exec jekyll serve. Your site should be available at http://localhost:4000.
5. Create a GitHub repository: Create a new GitHub repository and push your Jekyll site to the main branch.
6. Set up GitHub Pages: In your repository settings, enable GitHub Pages by selecting the main branch as the source.
7. Create a GitHub Actions Workflow (this is automatically down following the official tutorial): In your repository, create a .github/workflows/main.yml file and set up the workflow to build and deploy your site. Here's a sample workflow:
  ```yaml
  on:
    push:
      branches: [ "main" ]
    pull_request:
      branches: [ "main" ]

  jobs:
    build:

      runs-on: ubuntu-latest

      steps:
      - uses: actions/checkout@v3
      - name: Set up Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: 2.7.4
      - name: Install dependencies
        run: bundle install
      - name: Build the site
        run: bundle exec jekyll build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./_site
  ```
8. Commit and push: Commit and push the changes to the main branch. Your site will be built and deployed using the GitHub Actions Workflow.

## Bugs Encountered

- CSS not loading: The CSS files were not loading correctly due to an incorrect baseurl. This was fixed by updating the head.html file to include {{ site.baseurl }} in the stylesheet link.

- Missing navigation links: The navigation menu was not displaying the desired links. This was resolved by adding the site.baseurl in the header.html file, which fixed the navigation links. (remember add `--- layout title ---`)
