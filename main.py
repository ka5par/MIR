#import myller.extractor as me
import glob

import pop_music_highlighter.extractor as pmhe

if __name__ == '__main__':
    # fs = ['../data/Pink Floyd - The Great Gig in The Sky.wav']  # list
    fs = glob.glob("data/*.mp3")
    print(fs)

    # myller
    #me.extract(fs, length=10, save_SSM=True, save_thumbnail=True, save_wav=True, save_SP=True)

    # pop_music_highlighter
    pmhe.extract(fs, length=10, save_score=True, save_thumbnail=True, save_wav=True)
