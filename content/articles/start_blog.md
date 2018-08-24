Title: Start Your Data Science Blog by Pelican
Slug: start_blog
Date: 2018-03-01 12:00
Category: Others
Tags: Blog
Author: Tomoaki Fujii
Status: published


Blogging is one of the fantastic ways to demonstrate your projects and help you understand stuff in more depth. Especially, I believe that blogging helps you land a job more efficiently. Even if you are not looking for a new position, writing articles you are working on would be the practice to explain stuff to others, which always requires deep understanding. Indeed, A. Einstein mentioned
> If you can't explain it to a 6 year old then you really don't understand it yourself

Thus, blogging brings you a lot of benefits. Today is the date to start your blog!

In this article, I explain how to build your technology blog, especially data science blog.

I know that folks working around data science space hate suffering from stuff like learning HTML and making beautiful web design. I am one of them.
Then, static site generators comes in makes blogging simpler to even non professional guys like me. There are a few options for static site generators such as [Pelican](http://docs.getpelican.com/en/stable/) written in Python and [Jekyll](https://jekyllrb.com/) written in Ruby.

In daily analytics, I spend a lot of time on [Jupyter Notebook](http://jupyter.org/). So, my choice of the platform goes to [Pelican](http://docs.getpelican.com/en/stable/), which is able to generate articles directly from IPython Notebook file.

Let's dig into how to write articles with Pelican!

# Build the environment
We will go through the following processes:
1. Install Pelican
2. Create a default environment
3. Set up external plugins

### 1. Install Pelican
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

### 2. Create a default environment
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

Let's start from `pelicanconf.py`. As an example, what I am using is [here](https://github.com/jjakimoto/jjakimoto.github.io/blob/develop/pelicanconf.py).

If you check my `pelicanconf.py`, you may notice the part,
```python
THEME = "themes/mytheme"
```

This folder defines framework of your blog, and of course, it is customizable. My recommendation is finding some cool blog and arrange their them according to your demand. In my case, I add some extra stuff to [this theme](https://github.com/rossant/rossant.github.io/tree/sources/themes). The them what I amd using this website is [here]([here](https://github.com/jjakimoto/jjakimoto.github.io/tree/develop/themes/mytheme). You can see more detail at [the official documentation](http://docs.getpelican.com/en/3.6.3/themes.html) as to how to customize your theme.


### 3. Set up external plugins
Next, we are going to introduce external plugins for IPython Notebook and Markdown.

You can download files from [this repository](https://github.com/danielfrg/pelican-ipynb) for IPython Notebook and [this repository](https://github.com/getpelican/pelican-plugins) for Markdown.
The following commands introduce plugins in your repository
```buildoutcfg
# Install IPython plugins
git submodule add git://github.com/danielfrg/pelican-ipynb.git plugins/ipynb
# Install Pelican plugins
git submodule add git@github.com:getpelican/pelican-plugins.git
```


To activate these plubins, you should add the followings to your `pelicanconf.py`.
```
MARKUP = ('md', 'ipynb')
PLUGIN_PATHS = ['./plugins', './pelican-plugins']
PLUGINS = ['ipynb.markup', 'render_math', 'better_codeblock_line_numbering']
MD_EXTENSIONS = [
    'codehilite(css_class=highlight,linenums=False)',
    'extra'
    ]
```

#### All set for the environment!!!!!



# Set up GitHub repository
Pelican blog is managed by a GitHub repository. So, you need to create a repository for your blog with the following procedures under blog folder:

- Create a repository called `username.github.io`, where username is your Github username. In my case that is `jjakimoto.github.io`.

- Add the repository as a remote for your local git repository by running git remote add origin git@github.com:username/username.github.io.git -- replace both references to username with your Github username.

- Add the following line in publishconf.py:
```python
SITEURL = http://username.github.io
```
where username is your Github username.

- Run git checkout -b develop to create and switch to a branch called develop. We can't use master to store our notebooks, since that's the branch used by Github Pages.

- Create a commit and push to Github like normal (using git add, git commit, and git push).

We have set up the GitHub repository for publishing your article!
Next, we move on to how to write your articles.

# Write an article
When we writing your articles, we have two options in file format: Markdown `*.md` and IPython Notebook `*.ipynb`.

When you write an article from Markdown you always have to add meta information on top of the article. If you are writing article named `hoge.ipynb`, you have to make `hoge.nbdata' (hoge.ipynb-meta is previously used) and add meta information. In both cases, meta information looks like this:
```
Title: First Post
Slug: first-post
Date: 2018-03-01 12:00
Category: Blogs
Tags: Pelican, Data Science
author: Tomoaki Fujii
Summary: My first post, read it to find out.
Status: published
```

* Title -- the title of the post.
* Slug -- the path at which the post will be accessed on the server. For example, I set
```python
ARTICLE_URL = 'articles/{date:%Y}/{date:%b}/{date:%d}/{slug}/'
```
in my `pelicanconf.py`. So, your article can be accessed through `HOME_URL/articles/year/month/date/slug/`. In the above example, `https://jjakimoto.github.io/articles/2018/03/01/first-post/`.
* Date -- the date the post will be published.
* Category -- a category for the post -- this can be anything.
* Tags -- a space-separated list of tags to use for the post. These can be anything.
* Author -- the name of the author of the post.
* Summary -- a short summary of your post.
* Status -- if you set it `draft`, this article will not be added in index of your blog.  If you want to publish it, you should set to `published`.


Their default values can be set in `pelicanconf.py`. In my case, I set up
```python
DEFAULT_METADATA = {
    'status': 'draft',
}
```


# Generate page
We chose yes for
> Do you want to generate a Fabfile/Makefile to automate generation and publishing? (Y/n)

when executing `pelican-quickstart` in the previous section.
This generates Makefile and Fabfile for automating publication process.
I usually use Makefile for the publication with the following commands.

```
make html
```
This command generates HTML files according to the files under the content folder.

```
make serve
```
This command starts your blog on your local server. In the default setting, the blog will start on `http://localhost:8000`. This command is helpful when checking how the blog actually looks like before the publication.
If you have any drafts, they will be stored under `http://localhost:8000/drafts`.

For the publication, execute the following bash commands:
```console
git add -A
git commit -m"New publication"
git push origin develop# Update develop branch
pelican content -s publishconf.py
ghp-import output -b master
git push origin master
```

I write them in a file called [publish.sh](https://github.com/jjakimoto/jjakimoto.github.io/blob/develop/publish.sh).
Then, I just execute
```
bash publish.sh
```
for the publication.


That's it!!
Enjoy writing your blog. I hope blogging will help your aspiring career.

Thanks for reading ;)

I wrote this article in reference to the followings:

- [Setting up a blog with Pelican and GitHub Pages](http://cyrille.rossant.net/pelican-github/)

- [Building a data science portfolio: Making a data science blog](https://www.dataquest.io/blog/how-to-setup-a-data-science-blog/)

Check them out!
