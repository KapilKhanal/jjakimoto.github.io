Title: Start your data science blog by Pelican
Slug: start_blog
Date: 2018-02-28 12:00
Category: Others
Tags: Blog
Author: Tomoaki Fujii
Status: draft
Summary: Explain how to start your data science blog.


Blogging is one of the fantastic ways to demonstrate your projects and help you understand stuffs in more depth. Especially, I believe that blogging helps you land a job more efficiently. Even if you are not looking for a new position, writing articles you are working on would be the practice to explain stuffs to others, which always requires deep understanding. Indeed, A. Einstein mentioned
> If you can't explain it to a 6 year old then you really don't understand it yourself
Thus, blogging brings you a lot of benefits. Today is the date to start your blog!

In this article, I explain how to build your technology blog, especially data science blog.

I know that folks working around data science space hate suffering from stuffs like learning HTML and making beautiful web design. I am one of them.
Then, static site generators comes in makes blogging simpler to even non professional guys like me. There are a few options for static site generators such as [Pelican](http://docs.getpelican.com/en/stable/) written in Python and [Jekyll](https://jekyllrb.com/) written in Ruby.

In daily analytics, I spend a lot of time on [Jupyter Notebook](http://jupyter.org/). So, my choice of the platform goes to [Pelican](http://docs.getpelican.com/en/stable/), which is able to generate articles directly from IPython Notebook file.

Let's dig into how to write articles with Pelican!

# Build the environment
We will go through the following processes:
1. Install Pelican
2. Create a default environment
3. Set up jupyter extension

## 1. Install Pelican
Before installing anything, building virtual environment is recommended to avoid messing up your local python environment. We use `virtualenv` here and install it through pip.
```console
foo@bar:~$ pip install virtualenv
foo@bar:~$ mkdir ~/virtualenvs
foo@bar:~$ cd ~/virtualenvs
foo@bar:~$ virtualenv blogenv
foo@bar:~$ source virtualenvs/blogenv/bin/activate
(blogenv) foo@bar:~$
```
Now we have activated a virtual environment. Python packages we install from now does not affect your local python environment.

Let's install Pelican and Markdown over the established virtual environment through pip.
```console
pip install pelican markdown
```

## 2. Create a default environment
We determine under which folder the blog will be built. In this article, we are going to build the environment under '~/blog'.
Under this folder, we will make the following files:
* requirements.txt
* .gitignore

requirements.txt tells you what files is required to use your program.
Here is the example:
```
Markdown==2.6.11
pelican==3.7.1
jupyter>=1.0
ipython>=4.0
nbconvert>=4.0
bs4==4.6.0
ghp-import==0.4.1
matplotlib==2.0.2
```
Running `pip install -r requirements.txt` installs all their packages.


`.gitignore` avoids you annoyed to mess up git repository. The file whose name matched with patterns in this file will be ignored when executing git command.

Now, we are going to start your own blog.
In Pelican, there is command `pelican quick-start`. Its execution will get you the following console.

```
(blogenv) foo@bar:~$ pelican-quickstart
Welcome to pelican-quickstart v3.7.1.

This script will help you create a new Pelican-based website.

Please answer the following questions so this script can generate the files
needed by Pelican.


> Where do you want to create your new web site? [.]
> What will be the title of this web site? Data Rounder
> Who will be the author of this web site? Tomoaki Fujii
> What will be the default language of this web site? [en]
> Do you want to specify a URL prefix? e.g., http://example.com   (Y/n) n
> Do you want to enable article pagination? (Y/n)
> How many articles per page do you want? [10]
> What is your time zone? [Europe/Paris] America/New_York
> Do you want to generate a Fabfile/Makefile to automate generation and publishing? (Y/n)
> Do you want an auto-reload & simpleHTTP script to assist with theme and site development? (Y/n)
> Do you want to upload your website using FTP? (y/N)
> Do you want to upload your website using SSH? (y/N)
> Do you want to upload your website using Dropbox? (y/N)
> Do you want to upload your website using S3? (y/N)
> Do you want to upload your website using Rackspace Cloud Files? (y/N)
> Do you want to upload your website using GitHub Pages? (y/N)
```

After this process has finished without errors, we have the following files under `~/blog`:
```
content
pelicanconf.py
publishconf.py
fabfile.py
output
develop_server.sh
Makefile
```

Among files above, we edit `content` and `pelicanconf.py` frequently.

Let's start from `pelicanconf.py`. As an example, what I am using is posted here. This combines themes used in this site and some other couples of things.

If you check my `pelicanconf.py`, you may notice the part,
```python
THEME = "themes/mytheme"
```

This part defines how your blog looks like. This file is customizable. In my case, I add some stuffs based on [this theme](https://github.com/rossant/rossant.github.io/tree/sources/themes). You can see more detail as to how to customize your theme.

You also have to introduce external plugin files for IPython Notebook and Markdown.

For IPython Notebook, you can download files from [this repository](https://github.com/danielfrg/pelican-ipynb).

For Markdown, you need to download more generic plugins from [this repository](https://github.com/getpelican/pelican-plugins).

After install, you should add the followings to your `pelicanconf.py`.
```
MARKUP = ('md', 'ipynb')
PLUGIN_PATHS = ['./plugins', './pelican-plugins']
PLUGINS = ['ipynb.markup', 'render_math', 'better_codeblock_line_numbering']
MD_EXTENSIONS = [
    'codehilite(css_class=highlight,linenums=False)',
    'extra'
    ]
```

All  is set for `pelicanconf.py`. Next, we move on to creating actual articles.

Basically, you create articles under `content`.





























































I use [this site](http://cyrille.rossant.net/pelican-github/) and [this site](https://www.dataquest.io/blog/how-to-setup-a-data-science-blog/).
