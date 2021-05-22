from urllib.request import urlopen
from zipfile import ZipFile

zipurl = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
zipresp = urlopen(zipurl)
tempzip = open("/data/glove.zip", "wb")
tempzip.write(zipresp.read())
tempzip.close()
zf = ZipFile("/content/glove.zip")
    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
zf.extractall(path = 'data')
zf.close()