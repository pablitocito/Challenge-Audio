# Dataloader
from torch.utils.data import DataLoader, Dataset, random_split
from Utils import AudioUtil

class SoundDS(Dataset):

  def __init__(self, df):

    self.df = df
    self.duration = 10000 # Hyper-parametrisation
    self.sr = 44100 # Hyper-parametrisation
    self.channel = 2
    self.shift_pct = 0.4
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)    
  


  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    try:
      # Absolute file path of the audio file - concatenate the audio directory with
      # the relative path
      audio_file = self.df.loc[idx, 'mp3_filepath']

      # Get the release date
      year_id = self.df.loc[idx, 'year']

      aud = AudioUtil.open(audio_file)

      # Some sounds have a higher sample rate, or fewer channels compared to the
      # majority. So make all sounds have the same number of channels and same 
      # sample rate. Unless the sample rate is the same, the pad_trunc will still
      # result in arrays of different lengths, even though the sound duration is
      # the same.

      # Resample
      reaud = AudioUtil.resample(aud, self.sr)

      # Re-channel
      rechan = AudioUtil.rechannel(reaud, self.channel)

      # Size uniformisation
      dur_aud = AudioUtil.pad_trunc(rechan, self.duration)

      # Data augmentation
      shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)

      # Convert to a Mel-Spectrogram
      sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)


      # Data augmentation
      aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
      # Validate the augmented data
    
      return aug_sgram, year_id
    
    
    except Exception as e:
      print(f"Error processing data for index {idx}: {e} \n")
      return None 
    


if __name__ == '__main__':
  