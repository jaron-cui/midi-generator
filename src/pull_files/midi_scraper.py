import requests
import urllib.request
import os
from bs4 import BeautifulSoup, SoupStrainer
import random


def scrape(link, depth, links):
    url = link
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print_links(soup, depth, links)


def print_links(result, depth, links):
  for i in result.find_all('a', href=True):
    if i['href'].endswith('.htm') and not('classic.htm' in str(i['href'])) and depth:
      scrape("https://www.midiworld.com/" + i['href'], False, links)
    if i['href'].endswith('.mid'):
      links.append(i['href'])


def main():
  links = []
  scrape('https://www.midiworld.com/classic.htm', True, links)

  download = False
  print(len(links))
  # Construct the full path for the downloaded file
  folder_path = 'midi_files'
  os.makedirs(folder_path, exist_ok=True)
  i = 0
  # random.shuffle(links)

  for link in links:
    i+=1
    print(str(i), ":", link)
    if download:
      full_path = os.path.join(folder_path, link[link.rfind('/') + 1:])
      urllib.request.urlretrieve(link, full_path)


if __name__ == '__main__':
  main()
