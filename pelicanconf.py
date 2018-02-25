#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import os
from os.path import dirname, abspath, join

HOME_DIR = os.getenv("HOME")
CURRENT_DIR = dirname(abspath("__file__"))


# Name
AUTHOR = 'Tomoaki Fujii'
SITENAME = 'Data Rounder'
SITESUBTITLE = 'Machine Learning, Finance, and Technologies'
SITEURL = 'http://jjakimoto.github.io'
SIDEBAR_NAME = AUTHOR
SIDEBAR_EMAIL = "f.j.akimoto@gmail.com"
SIDEBAR_TAGS = ['Machine Learning',
                'Deep Learning',
                'Finance',
                'Python',
                ]
MENUITEMS = [('Home', '/'),
             ('Books', '/pages/books/'),
             ('About', '/pages/about/'),
             ]

# pusblish an article as a draft in default setting
DEFAULT_METADATA = {
    'status': 'draft',
}

# Basic settings
DEFAULT_CATEGORY = 'misc'
DISPLAY_CATEGORIES_ON_MENU = True
DISPLAY_PAGES_ON_MENU = True
IGNORE_FILES = ['.#*']
MARKDOWN = {}

# PATH settings
OUTPUT_PATH = 'output/'
PATH = 'content'
STATIC_PATHS = ['images']
PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['articles']

# Extention
MARKUP = ('md', 'ipynb')
PLUGIN_PATHS = ['./plugins', "./pelican-plugins"]
PLUGINS = ['ipynb.markup', "render_math"]

# URL settings
ARTICLE_URL = 'articles/{date:%Y}/{date:%b}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'articles/{date:%Y}/{date:%b}/{date:%d}/{slug}/index.html'
PAGE_URL = 'pages/{slug}/'
PAGE_SAVE_AS = 'pages/{slug}/index.html'

# Time and Date settings
TIMEZONE = 'America/New_York'
DEFAULT_DATE_FORMATS = '%a, %m/%d/%Y'
LOCALE = ('en_US')

# Template pages settings
TEMPLATE_PAGES = None

THEME = "themes/mytheme1"

# PELICAN_COMMENT_SYSTEM = True
# PELICAN_COMMENT_SYSTEM_IDENTICON_DATA = ('author',)
DISQUS_SITENAME = "datarounder"


DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'
# TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = 'feeds/%s.atom.xml'
# AUTHOR_FEED_RSS = None

# Blogroll
# LINKS = (('GitHub', 'https://github.com/jjakimoto'),)

# GITHUB_URL = 'https://github.com/jjakimoto'

# Social widget
SOCIAL = (('GitHub', 'https://github.com/jjakimoto'),
          ('LinkedIn', 'https://www.linkedin.com/in/tomoaki-fujii-5bba56133'),
          ('Facebook', 'https://www.facebook.com/tomoaki.fujii.73'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True


# COMMENT_URL = "#my_own_comment_id_{slug}"
