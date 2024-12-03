from bing_image_downloader import downloader
downloader.download('nissan march 2017', limit=100,  output_dir='dataset/nissan2017', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)