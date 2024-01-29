import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt


class Audio:
    '''
    Easily record sound for X seconds.
    Handle recorded sound data as numpy array.
    Save data as a .wav file.
    Plot data.
        Save plot as .png at specified location.
        Default location is wherever stuff is being run.

    Conceptually, there should only be 1 stream open at a time.
    Assuming that you will always start this by recording some audio.
    '''

    def __init__(self):

        self.data = None
        
        self.noise_floor = None

        self._pyaudio_obj = None
        self._stream = None

        self.audio_params = None

    def _open_stream(
        self,
        rate_hz: int=16000,
        channels: int=1,
        frames_per_buffer: int=1024,
        audio_format: int=pyaudio.paInt16) -> None:
        '''
        Opens an audio stream for recording using pyaudio.

        channels: 1 is mono, 2 is stereo.
        rate_hz: Recording rate - 16k is pretty standard for a lot of applications. 
        audio_format: TODO. 
        frames_per_buffer: parameter for managing the trade-off between audio processing
            latency and overhead. When audio data is read from or written to the stream, 
            it is done in chunks of this buffer size. A smaller buffer size means that 
            audio data is transferred more frequently with lower latency but at potentially
            higher processing overhead, as the system has to handle more buffers per unit time.
            Conversely, a larger buffer size reduces the processing overhead (since there 
            are fewer buffers to handle) but increases the latency, which might not be 
            suitable for real-time audio processing tasks.
        '''

        self._pyaudio_obj = pyaudio.PyAudio()

        self._stream = self._pyaudio_obj.open(
            input=True,  # Tells the stream you are opening to record data.
            rate=rate_hz,  # Sample Rate.
            channels=channels,  # If not set, this should default to the recording device's default.
            frames_per_buffer=frames_per_buffer,
            format=audio_format)
        
        # Create a params structure so that even after the audio is done
        # recording and the stream is closed, these properties will be available.
        self.audio_params = {
            'rate_hz': rate_hz,
            'channels': channels,
            'format': audio_format}
        
        return
    
    def _close_stream(self) -> None:
        '''
        Cleanly close the stream and the PyAudio object from which the stream originated.
        '''
        
        # Cleanup stream and audio object.
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio_obj.terminate()

        return

    def _read_stream(self, read_time_s: float=.5) -> np.array:
        '''
        Reads an open audio stream to record and store sound levels into a numpy array.

        read_time_s: the total time to read the stream in seconds.
        frame_count: The chunk size of how many frames to read at a time. It's possible that this is
            somewhat dependent on the audio stream which is initialized with a "frames_per_buffer"
            parameter. I haven't tested trying to open an audio stream with one type of frame count
            and then reading with a different. That's a TODO.
        '''

        # Given stream collection rate and total time needed, calculate frames to read.
        frames_to_read = int(self._stream._rate * read_time_s)

        raw_sample = self._stream.read(frames_to_read)
        numeric_sample = np.double(np.frombuffer(raw_sample, dtype=np.int16))  # Converts some binary buffer obj to numpy array.

        return numeric_sample
    
    def detect_audio():
        '''
        Opens an audio stream and tries to collect the next full sample of audio.
        Automatically cuts off the stream once too much time has passed.
        Limits the collected audio to the start and stop of activity.
        Does some calculation to only record and keep audio if it counts as activity.
        '''

        return

    def record(self, time_s: float):
        '''
        Record an audio sample for X seconds. Saves recorded sample into self.data.

        time_s: time to record sample for in seconds.
        '''

        self._open_stream()
        self.data = self._read_stream(read_time_s=time_s)
        self._close_stream()

        return

    def _calc_activity_levels(self, audio_array: np.array):
        '''
        Determine if an audio sample has any activity, and if it ends in a dead zone. 
        '''

        SOUND_THRESHOLD = 1000  # Anything below this is considered background noise. Above is considered "active".
        PERCENT_ACTIVE_REQUIRED = 10  # This percent samples over the SOUND_THRESHOLD is considered an "active" clip. 
        PERCENT_LOOK_BACK = 25  # If there no above-noise-threshold elements in this "back percent" of the sample, its the end of the clip.

        is_active = False  # Initial state for sample has active points.
        ends_dead = True  # Initial for the clip ending in a non-active state.

        sample_length = len(audio_array)

        n_look_back = int(np.round(PERCENT_LOOK_BACK/100 * sample_length))  # Number of frames to look back on.
        num_active_required = int(np.round(PERCENT_ACTIVE_REQUIRED/100 * sample_length))  # Num active frames required.

        # This is a list of the all the indexes of the frames that were considered "active".
        i_active = [i for i, value in enumerate(audio_array) if abs(value) > SOUND_THRESHOLD]

        if len(i_active) >= num_active_required:  # Total number of active frames is above threshold...
            is_active = True
        
            # If the last active frame is later than the percent look back frame...
            if i_active[-1] > (sample_length - n_look_back):  # Last active value.
                ends_dead = False

        return is_active, ends_dead
    
    def calc_noise_floor(self, time_s: float=3.0) -> float:
        '''
        Read in a short audio clip and try to determine the current level of background noise.
        This will be helpful for other methods which are trying to determine the activity level
        of a sound clip.

        This is meant to be run when there is only "ambient" background noise present and not
        anything active (like someone talking).

        time_s: time to record in seconds to calculate average noise.
        '''

        self.record(time_s=time_s)

        return
    
    def plot(self):
        '''
        Plots current data. Since plot objects tend to freeze up the system until they are closed, 
        plot will create a plot and save it as a .png without ever actually showing it to screen. 
        '''

        if self.data is None:
            raise ValueError('Must have data to plot!')
        
        time_axis = np.arange(len(self.data)) / sample_rate

        # Plotting
        plt.figure(figsize=(10, 4))  # Optional: Adjust the figure size
        plt.plot(time_axis, self.data)
        plt.title("Audio Signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Save the plot as a PNG file
        plt.savefig("audio_signal.png")
    
    def save_wav(self, save_path: str='audio.wav'):
        '''
        Saves the current data to a .wav file.

        save_path: full file path to .wav file in save location.
        '''
        
        if self.data is None:
            raise ValueError('Must have data to save as .wav file!')
        if save_path is not None and not save_path.endswith('.wav'):
            raise ValueError('"save_path" must be a .wav file!')
        
        byte_data = self.data.tobytes()  # Convert to bytes.

        wav_data = wave.open(save_path, 'wb')
        wav_data.setnchannels(self._stream._channels)
        wav_data.setsampwidth(self._pyaudio_obj.get_sample_size(self._stream._format))
        wav_data.setframerate(self._stream._rate)
        # wav_data.writeframes(b''.join(frames))
        wav_data.writeframes(byte_data)
        wav_data.close()

        return
