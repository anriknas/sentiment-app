import urllib.request
import sys
import zipfile

url = 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'

def download_url(url, save_path):
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())


if __name__ == "__main__":
    download_url(url,'../data.zip')
    with zipfile.ZipFile('../data.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/')