Title: Start your data science blog by Pelican
Slug: start_blog
Date: 2018-02-28 12:00
Category: General
Tags: Blog
Author: Tomoaki Fujii
Status: draft
Summary: Explain how to start your data science blog.



Blogging can be a fantastic way to demonstrate your skills, learn topics in more depth, and build an audience. There are quite a few examples of data science and programming blogs that have helped their authors land jobs or make important connections. Blogging is one of the most important things that any aspiring programmer or data scientist should be doing on a regular basis.

Unfortunately, one very arbitrary barrier to blogging can be knowing how to setup a blog in the first place. In this post, we'll cover how to create a blog using Python, how to create posts using Jupyter notebook, and how to deploy the blog live using Github Pages. After reading this post, you'll be able to create your own data science blog, and author posts in a familiar and simple interface.


Blogging is one of the fantastic ways to demonstrate your projects and sum up what you have learned. More importantly, blogging brings you the following benefits:

1. Enhance your skills
2. good self advertisement
3. Understand subjects in more depth

1. When you post some articles, you may get comments, which may be hash one. This may give you the direction as to how you improve your projects or skills.
2. Having your technology blog is effective way to advertise you to the outside
3. Explaining something to someone else always requires you to understand the subject deeply.
"If you can't explain it to a 6 year old then you really don't understand it yourself" - A. Einstein

, which leads to the good self-advertisement. skills and learn topics in more depth. Considering that people in technology space spend a lot of time collecting information form the Internet, it is effective way for aspiring software engineers to build the audience. Blogging brings you the following benefits:
1. learn a lot


In this article, I explain how to build your technology, especially data science, blog.
In daily analytics part, I spend a lot of time on [jupyter notebook](http://jupyter.org/). So, my choice of the platform goes to [Pelican](http://docs.getpelican.com/en/stable/). Pelican allows you to generate HTML article directly from your notebook file, filename.ipynb.

There are many options for static site generators such as Pelican,


# Build the environment
We go through the following procedures:
1. Install Pelican
2. Create default environment
3. Set up jupyter extension

## 1. Install Pelican
Before installing anything, building virtual environment is recommended to avoid incompatibility with your machine settings.
```console
foo@bar:~$ whoami
foo
```

Create a folder -- we'll put our blog content and styles in this folder. We'll refer to it in this tutorial as jupyter-blog, but you can call it whatever you want.
cd into jupyter-blog.
Create a file called .gitignore, and add in the content from this file. We'll need to eventually commit our repo to git, and this will exclude some files when we do.
Create and activate a virtual environment.
Create a file called requirements.txt in jupyter-blog, with the following content:
Markdown==2.6.6
pelican==3.6.3
jupyter>=1.0
ipython>=4.0
nbconvert>=4.0
beautifulsoup4
ghp-import==0.4.1
matplotlib==1.5.1
Run pip install -r requirements.txt in jupyter-blog to install all of the packages in requirements.txt.
