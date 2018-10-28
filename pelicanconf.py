#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals


THEME = "themes/mytheme"

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
             ('Links', '/pages/links/'),
             ]

# pusblish an article as a draft in default setting
DEFAULT_METADATA = {
    'status': 'draft',
}

# Basic settings
DEFAULT_CATEGORY = 'misc'
DISPLAY_CATEGORIES_ON_MENU = True
DISPLAY_PAGES_ON_MENU = True
IGNORE_FILES = ['.#*', '.ipynb_checkpoints']
MARKDOWN = {}

# PATH settings
OUTPUT_PATH = 'output/'
PATH = 'content'
STATIC_PATHS = ['images', 'data', 'publications']
PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['articles']

# Extention
MARKUP = ('md', 'ipynb')
PLUGIN_PATHS = ['./pelican-plugins', './plugins']
PLUGINS = ['render_math', 'ipynb.markup', 'better_codeblock_line_numbering']

MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
    },
    'output_format': 'html5',
}

# URL settings
ARTICLE_URL = 'articles/{slug}/'
ARTICLE_SAVE_AS = 'articles/{slug}/index.html'
PAGE_URL = 'pages/{slug}/'
PAGE_SAVE_AS = 'pages/{slug}/index.html'

# Time and Date settings
TIMEZONE = 'America/New_York'
DEFAULT_DATE_FORMATS = '%a, %m/%d/%Y'
# LOCALE = ('en_US')

# Template pages settings
TEMPLATE_PAGES = None

DISQUS_SITENAME = "datarounder"


DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'
# TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = 'feeds/%s.atom.xml'
# AUTHOR_FEED_RSS = None


# Social widget
SOCIAL = (('GitHub', 'https://github.com/jjakimoto'),
          ('LinkedIn', 'https://www.linkedin.com/in/tomoaki-fujii-5bba56133'),
          ('Facebook', 'https://www.facebook.com/tomoaki.fujii.73'),)

DEFAULT_PAGINATION = 10

