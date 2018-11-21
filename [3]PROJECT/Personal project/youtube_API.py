import httplib2
import os
import sys
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

CLIENT_SECRETS_FILE = "client_secret.json"

YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

MISSING_CLIENT_SECRETS_MESSAGE = "WARNING: Please configure OAuth 2.0"

def get_authenticated_service(args):
    flow = flow_from_clientsecrets(
        CLIENT_SECRETS_FILE,
        scope=YOUTUBE_READ_WRITE_SSL_SCOPE,
        message=MISSING_CLIENT_SECRETS_MESSAGE)

    storage = Storage("youtube-api-snippets-oauth2.json")
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)
    return build(
        API_SERVICE_NAME,
        API_VERSION,
        http=credentials.authorize(httplib2.Http()))


args = argparser.parse_args()
service = get_authenticated_service(args)

def print_results(results):
    print(results)

def build_resource(properties):
    resource = {}
    for p in properties:
        prop_array = p.split('.')
        ref = resource
        for pa in range(0, len(prop_array)):
            is_array = False
            key = prop_array[pa]
            if key[-2:] == '[]':
                key = key[0:len(key) - 2:]
                is_array = True
            if pa == (len(prop_array) - 1):
                if properties[p]:
                    if is_array:
                        ref[key] = properties[p].split(',')
                    else:
                        ref[key] = properties[p]
            elif key not in ref:
                ref[key] = {}
                ref = ref[key]
            else:

                ref = ref[key]
    return resource


def remove_empty_kwargs(**kwargs):
    good_kwargs = {}
    if kwargs is not None:
        for key, value in kwargs.items():
            if value:
                good_kwargs[key] = value
    return good_kwargs


def comment_threads_list_by_video_id(service, **kwargs):
    kwargs = remove_empty_kwargs(**kwargs)  # See full sample for function
    results = service.commentThreads().list(**kwargs).execute()
    # print_results(results)
    return results

rawdata = comment_threads_list_by_video_id(service, part='snippet,replies', videoId='cCru-rXbBjs')

data = []
for item in rawdata['items']:
    parents = {}
    itemP = item['snippet']['topLevelComment']
    parents['id'] = itemP['id']
    parents['author'] = itemP['snippet']['authorDisplayName']
    parents['time'] = itemP['snippet']['updatedAt']
    parents['text'] = itemP['snippet']['textDisplay']
    subdata = []
    if not 'replies' in item.keys():
        continue
    for comment in item['replies']['comments']:
        children = {}
        children['id'] = comment['id']
        children['author'] = comment['snippet']['authorDisplayName']
        children['time'] = comment['snippet']['updatedAt']
        children['text'] = comment['snippet']['textDisplay']
        subdata.append(children)
        parents['comments'] = subdata
        data.append(parents)

print(rawdata['items'])