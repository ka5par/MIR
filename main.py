import myller.extractor as me
import pop_music_highlighter.extractor as pmhe

if __name__ == '__main__':
    fs = ['data/Pink Floyd - The Great Gig in The Sky.wav']  # list
    # fs = ["data/Pink Floyd - The Great Gig in The Sky.wav", "data/FMP_C4_Audio_Beatles_YouCantDoThat.wav"]

    # myller
    me.extract(fs, length=10, save_SSM=True, save_thumbnail=True, save_wav=True, save_SP=True, output_path='output/repetition/')

    # pop_music_highlighter
    pmhe.extract(fs, length=10, save_score=True, save_thumbnail=True, save_wav=True, output_path='output/attention/')
