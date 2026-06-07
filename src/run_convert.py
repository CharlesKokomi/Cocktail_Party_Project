from convert_audio import convert_flac_to_wav

root_directory = r"F:\CodeSpace\Cocktail_Party_Project\data\LibriSpeech"
target_directory = r"F:\CodeSpace\Cocktail_Party_Project\data\processed_wav"

# 执行转换
print("开始转换音频，请耐心等待...")
convert_flac_to_wav(root_directory, target_directory)
print("转换全部完成！")