# LiveSpeechToText
Live speech to text transcription. Ambient sound is recorded and parsed into standalone words or phrases that are streamed to a voice model that will transcribe the sound to text. Works completely offline - no streaming to a service or external connections to websites or 3rd party services needed.

# Running:
tests/test_audio_object.py has a good run through of the Audio class and how to use it's various methods.

# Requirements:

## Requires Python 3.11.1 to work.
Other versions may work but not guranteed.

Create a virtual environment. Then install from "requirements.txt".
```
pip install -r requirements.txt
```

### TODO: Add notes about what kind of models from Hugging Face will work based on the system requirements. 
For instance, the Facebook960hr model seems to be too big (94.4 million parameters) for the Raspberry Pi 4B in it's default configuration, but runs fine on a decent desktop PC.

## Linux:
On linux (RPi), you might have to run:
```
sudo apt install portaudio19-dev python3-pyaudio
```

This should fix an error when installing from requirements.txt that yells about portaudio installing when trying to install "PyAudio" which is a library in requirements.txt that is needed.

I believe you will also have to install ffmpeg.
```
sudo install ffmpeg
```

### Setup Recording Device:
Select the speaker HAT by running:
```
alsamixer
```
and choosing the right soundcard for recording device using the F keys and menu options.

### TODO: Add instructions for recording and then playing a test .wav file.
This would be using arecord.

## Windows:
On Windows, there are a couple of pieces of functionality that won't work because they require the installation of another application. The only one I'm tracking is that Windows doesn't have the "ffmpeg" package. This means you wont be able to transcribe a sound file directly (i.e. point the model to a .flac and get a transcription) unless you have ffmpeg installed for Windows. All other functionality should still work.

---

### TODO: Breakout the instructions for enabling and using recording devices on the RPi to the HowToRaspberryPi Wiki.

### Enabling Recording Device Hardware:
The following information generated from Chat GPT4.

**User**: *I have a raspberry pi 4B.  I added a Seeed Studio 2 speaker mic HAT to the raspberry pi.  How do I enable the HAT on the pi and what config files do I edit to make this HAT the default recording hardware for the Raspberry Pi?*

**Chat GPT4:** To enable the Seeed Studio 2-Mic HAT on your Raspberry Pi 4B and set it as the default recording hardware, you'll need to follow a series of steps. These steps typically involve installing necessary drivers, configuring audio settings, and editing certain configuration files. Here's a general guide to get you started:

1. **Update Your Raspberry Pi:**
   First, make sure your Raspberry Pi is up to date. Open a terminal and run:
   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   ```

2. **Install the Seeed Studio 2-Mic HAT drivers:**
   Seeed Studio usually provides a script or commands to install the necessary drivers for their HATs. You can find these instructions on their website or in the documentation that came with the HAT. The installation command is generally something like this:
   ```bash
   git clone https://github.com/respeaker/seeed-voicecard.git
   cd seeed-voicecard
   sudo ./install.sh
   sudo reboot
   ```

   **USER NOTE:** For this block of code to install the drivers, see the section below on Seeed Hardware Setup instead. I think that driver set is more up to date than GPT's suggestion. Both would probably work.

3. **Verify the Installation:**
   After rebooting, check if the HAT is recognized by running:
   ```bash
   arecord -l
   ```
   This command lists all recording hardware. You should see the Seeed 2-Mic HAT listed.

4. **Set the 2-Mic HAT as the Default Recording Device:**
   To make the 2-Mic HAT the default recording device, you'll need to edit the `.asoundrc` file in your home directory. If this file doesn't exist, you'll create it.
   ```bash
   nano ~/.asoundrc
   ```
   Then, add the following configuration:
   ```bash
   pcm.!default {
       type asym
       playback.pcm {
           type plug
           slave.pcm "hw:0,0"
       }
       capture.pcm {
           type plug
           slave.pcm "hw:1,0"  # Adjust the card and device number based on your arecord -l output
       }
   }
   ```

5. **Adjust the ALSA Mixer Controls:**
   Use `alsamixer` to adjust the volume and capture settings:
   ```bash
   alsamixer
   ```
   Make sure to select the Seeed 2-Mic HAT and adjust the settings as needed.

6. **Test the Recording:**
   To test if everything is set up correctly, you can record a short audio clip:
   ```bash
   arecord -D plughw:1,0 -d 5 test.wav
   ```
   Replace `1,0` with the card and device number of your 2-Mic HAT if different. The `-d 5` option specifies a 5-second recording. Play the recording with:
   ```bash
   aplay test.wav
   ```

7. **Additional Configuration:**
   If you are using applications that rely on PulseAudio, you might need to configure PulseAudio to use the HAT as the default device. This can be done by editing the `/etc/pulse/default.pa` file.

Remember, these instructions are quite general. For specific details, especially regarding the driver installation for the Seeed Studio 2-Mic HAT, it's important to refer to the official documentation provided by Seeed Studio. Some steps might slightly vary depending on the specific version of the Raspberry Pi OS and the HAT firmware.

**User:** *Instead of editing the .asoundrc file, can I edit the "asound.config" file?  What is the difference between the ".asoundrc" file and the "asound.config" file?*

Yes, you can edit the `asound.conf` file instead of the `.asoundrc` file, and it's important to understand the difference between these two.

1. **`.asoundrc`:**
   - The `.asoundrc` file is a user-specific ALSA (Advanced Linux Sound Architecture) configuration file. It's typically located in a user's home directory (`~/.asoundrc`).
   - This file is read by ALSA for the particular user who owns the file. It means the configurations set in `.asoundrc` apply only to the user account under which it is saved.
   - If each user on the system requires different ALSA settings, `.asoundrc` is used to set these preferences on a per-user basis.

2. **`asound.conf`:**
   - The `asound.conf` file serves a similar purpose but is system-wide. It is typically located in `/etc/` directory (`/etc/asound.conf`).
   - This file is read by ALSA for all users on the system. Any settings defined here apply globally, across all user accounts.
   - If you want to set default ALSA configurations that apply to all users on the system, you use `asound.conf`.

**Modifying `asound.conf` for the Seeed Studio 2-Mic HAT:**

To set the Seeed Studio 2-Mic HAT as the default recording device for the entire system, you can edit the `asound.conf` file. The process is similar to editing `.asoundrc`, but with system-wide implications:

1. Open the `asound.conf` file with a text editor (you'll need sudo privileges):
   ```bash
   sudo nano /etc/asound.conf
   ```

2. Add the configuration for the Seeed Studio 2-Mic HAT, similar to what you would add in `.asoundrc`:
   ```bash
   pcm.!default {
       type asym
       playback.pcm {
           type plug
           slave.pcm "hw:0,0"
       }
       capture.pcm {
           type plug
           slave.pcm "hw:1,0"  # Adjust as necessary based on your system
       }
   }
   ```

   **USER NOTE:** The original settings for "slave.pcm" for "playback" and "capture" were originally "dmixed" and "array" respectively.

3. Save the file and exit the editor. You may need to restart the ALSA service or reboot the Raspberry Pi to apply these changes.

By editing `asound.conf`, you ensure that the Seeed Studio 2-Mic HAT is set as the default recording device for all users on your Raspberry Pi. Remember to always backup original configuration files before making changes.

---

**Chat GPT:** To ensure that your user account has the necessary permissions to access the audio hardware on your Raspberry Pi and to correctly configure your Python script to use the proper ALSA devices using PyAudio, follow these steps:

### Verifying User Permissions for Audio Hardware

1. **Check if Your User is in the Audio Group:**
   - Open a terminal and run the following command to see which groups your user belongs to:
     ```bash
     groups
     ```
   - Look for `audio` in the list. If it's not there, you need to add your user to the audio group.

2. **Add User to the Audio Group:**
   - If your user is not in the audio group, add them using this command:
     ```bash
     sudo usermod -a -G audio <username>
     ```
   - Replace `<username>` with your actual username.
   - Log out and log back in for the changes to take effect.

### Configuring PyAudio in Python Script

PyAudio is a Python library that interfaces with PortAudio for audio I/O. To use it for specifying input and output devices, you'll first need to make sure PyAudio is installed:

```bash
pip install pyaudio
```

#### Using PyAudio to Specify Devices

1. **List Available Devices:**
   - Use PyAudio to list all available audio devices. This will help you identify the device indices for your Seeed Studio 2-Mic HAT.

   ```python
   import pyaudio

   p = pyaudio.PyAudio()
   for i in range(p.get_device_count()):
       print(p.get_device_info_by_index(i))
   p.terminate()
   ```

2. **Identify the Device Indices:**
   - Run the above script to list all devices. Look for your Seeed Studio 2-Mic HAT in the output and note its input and output indices.

3. **Specify Devices in Your Script:**
   - When setting up the stream in PyAudio, specify the input and output device indices. For example:

   ```python
   import pyaudio

   # Set these variables to the correct device indices
   input_device_index = 2  # Replace with your input device index
   output_device_index = 3  # Replace with your output device index

   p = pyaudio.PyAudio()

   # For Recording
   stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=44100,
                   input=True,
                   input_device_index=input_device_index,
                   frames_per_buffer=1024)

   # For Playback
   stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=44100,
                   output=True,
                   output_device_index=output_device_index,
                   frames_per_buffer=1024)
   ```

   - In this example, replace `input_device_index` and `output_device_index` with the actual indices of your Seeed Studio 2-Mic HAT.

4. **Record and Play Audio:**
   - Use the `stream` object to read data for recording and write data for playback.
   - Here's an example of how you might record a short clip and then play it back:

   ```python
   # Record
   frames = []
   for _ in range(0, int(44100 / 1024 * 2)):  # Record for 2 seconds
       data = stream.read(1024)
       frames.append(data)

   # Stop Recording
   stream.stop_stream()
   stream.close()
   p.terminate()

   # Playback
   p = pyaudio.PyAudio()
   stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=44100,
                   output=True,
                   output_device_index=output_device_index)

   for frame in frames:
       stream.write(frame)

   # Stop Playback
   stream.stop_stream()
   stream.close()
   p.terminate()
   ```

By following these steps, you should be able to verify the necessary permissions and correctly configure your Python script to use the specified ALSA devices with PyAudio for both input and output.

---

Currently getting some issues with PyAudio. Specifically getting a bunch of non-breaking errors/warnings about "ALSA lib ... capture slave is not defined" and other ALSA warnings and some "jack server ..." warnings.

These all start on the command:
```
import pyaudio

p = pyaudio.PyAudio()
```

**Chat GPT:** If the errors occur specifically when you execute `p = pyaudio.PyAudio()`, it indicates that the issue is likely related to the initialization of the PyAudio system and its interaction with the underlying audio drivers or configurations. This step in PyAudio attempts to interface with the ALSA backend (or whichever audio backend is configured on your system), so any misconfigurations or compatibility issues at this level can lead to errors.

Here are some steps to troubleshoot and potentially resolve this issue:

### 1. Verify PyAudio and PortAudio Installation:

- Ensure that PyAudio and its underlying library, PortAudio, are correctly installed. PyAudio is a Python binding for PortAudio, so any issues with PortAudio could manifest as errors in PyAudio.
- You might want to reinstall or update them:
  ```bash
  pip uninstall pyaudio
  sudo apt-get install portaudio19-dev
  pip install pyaudio
  ```

### 2. Check ALSA Configuration:

- As previously discussed, verify that your ALSA configuration (`asound.conf` or `.asoundrc`) is correctly set up, especially if your script is the first thing that interacts with the audio system after booting.
- Test your audio setup with ALSA tools (`arecord`, `aplay`) to ensure that the configuration is working outside of PyAudio.

### 3. Test with Different Audio Devices:

- Sometimes, specifying a different audio device in the ALSA configuration can help identify if the issue is with a specific device or the overall setup.
- If possible, try testing with a simple USB microphone or speakers to see if PyAudio initializes correctly with a different audio setup.

### 4. Explore PyAudio Initialization Options:

- PyAudio can sometimes be finicky with device initialization. You might try initializing it without immediately opening a stream:
  ```python
  import pyaudio
  p = pyaudio.PyAudio()
  p.terminate()  # Close it immediately to test initialization
  ```

### 5. Check for Conflicts with Other Audio Servers:

- Ensure that there are no conflicts with other audio servers like PulseAudio or JACK. If you have multiple audio servers running, they might interfere with PyAudio's ability to access the ALSA backend.

### 6. Look for System Logs:

- Check system logs for any additional information about the errors. The output from `dmesg` or the contents of `/var/log/syslog` might provide more clues.

### 7. Python Environment:

- Consider the Python environment you're using. Are you in a virtual environment, or is it the system-wide Python installation? Sometimes, library paths and dependencies can behave differently in different environments.

### 8. Seek Specific Support:

- If these steps don't resolve the issue, consider seeking support from the PyAudio community or the specific forums for the Seeed Studio 2-Mic HAT. They might have encountered similar issues and could have specific advice.

### 9. Alternative Libraries:

- If PyAudio continues to be problematic, you might explore alternative Python libraries for audio input and output that are compatible with your hardware, like `sounddevice` or `python-soundfile`.

These troubleshooting steps are aimed at isolating and resolving the issue with PyAudio initialization. The exact solution may depend on the specific configuration and environment of your Raspberry Pi.

# Technique:

## 1. Stream sound samples to model

* Record super short samples of sound (~.1 seconds in length).
* Analyze each sample for exceeding a sound threshold above an adjustable limit.
* Continuously build a buffer of sound samples that contain sound above the threshold.
* Once you have a sample where the threshold is NOT exceeded, assume that this is the end of the phrase or word and stop building the buffer.
* Send the whole buffer to the model directly to be transcribed into text.
* Go back to listening for sound.

# Seeed Studio Microphone Hat Setup:
The Hin Tak repo seems to still be making current changes and support for the Respeaker Sound Cards (by Seeed Studio). I recommend the instructions on their [repo](https://github.com/HinTak/seeed-voicecard).

As a secondary, Adafruit has instructions for the sound card they sell which seems to be the same basic card (at least physically) and their [instructions](https://learn.adafruit.com/adafruit-voice-bonnet/audio-setup) also look the same (because I think it's the same drivers which supports the idea that the sound cards are the same).

[Instructions from Seeed](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT_Raspberry/) are the third resource for posterity and information but the last time their GitHub was updated was 2021 which is not encouraging.

Connect the HAT to the RPi. Make sure that the HAT is connected properly in orientation and that the pins are all lined up (it's suprisingly easy to be offset by a pin or two). **Make sure that the HAT is powered** (not from the RPi, but from another external power source, like where the RPi draws power from).

### Install the HAT drivers.
```
git clone https://github.com/HinTak/seeed-voicecard
cd seeed-voicecard
sudo ./install.sh
sudo reboot now
```

Once rebooted, check to make sure the HAT shows up. This should list all available sound recording devices.
```
arecord -l
```

You should see something like "seeed2micvoicec" listed under the recording devices.

# Notes:
