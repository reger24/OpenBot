package org.openbot.env.voicerecog;

import android.Manifest;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder.AudioSource;
import android.os.Binder;
import android.os.IBinder;
import android.widget.Toast;
import androidx.core.app.ActivityCompat;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import org.openbot.env.IDataReceived;
import org.openbot.env.SharedPreferencesManager;
import org.tensorflow.lite.Interpreter;
import timber.log.Timber;

/**
 * A service that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words. The code is derived from
 * org.tensorflow.lite.examples.speech.SpeechActivity adjusted to work as service
 *
 * @see <a
 *     href="https://github.com/tensorflow/examples/blob/189c662d500c20b9f9e93fef0af97d5311e64377/lite/examples/speech_commands/android/app/src/main/java/org/tensorflow/lite/examples/speech/SpeechActivity.java">SpeechActivity</a>
 */
public class VoiceRecognitionService extends Service {

  public static final String CMD_START_LISTEN = "start_listen";
  public static final String CMD_STOP_LISTEN = "stop_listen";
  // Constants that control the behavior of the recognition code and model
  // settings. See the audio recognition tutorial for a detailed explanation of
  // all these, but you should customize them to match your training settings if
  // you are running your own model.
  private static final int SAMPLE_RATE = 16000;
  private static final int SAMPLE_DURATION_MS = 1000;
  private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
  private static final long AVERAGE_WINDOW_DURATION_MS = 1000;
  private static final float DETECTION_THRESHOLD = 0.50f;
  private static final int SUPPRESSION_MS = 1500;
  private static final int MINIMUM_COUNT = 3;
  private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
  private static final String LABEL_FILENAME = /*"file:///android_asset/"*/
      "networks/voice/conv_actions_labels.txt";
  private static final String MODEL_FILENAME = /*"file:///android_asset/"*/
      "networks/voice/conv_actions_frozen.tflite";

  // Working variables.
  short[] recordingBuffer = new short[RECORDING_LENGTH];
  int recordingOffset = 0;
  boolean shouldContinue = true;
  private Thread recordingThread;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  private final ReentrantLock recordingBufferLock = new ReentrantLock();
  private final ReentrantLock tfLiteLock = new ReentrantLock();

  private List<String> labels = new ArrayList<String>();
  private RecognizeCommands recognizeCommands = null;
  private final Interpreter.Options tfLiteOptions = new Interpreter.Options();
  private MappedByteBuffer tfLiteModel;
  private Interpreter tfLite;
  private IDataReceived dataReceivedCallback; // callback to notify on new recognized command
  private AudioManager audioManager; // use to enable Bluetooth headset as audio source
  private boolean btHeadsetConnected = false; // Bluetooth headset status

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  private final IBinder binder = new LocalBinder();

  public class LocalBinder extends Binder {

    public VoiceRecognitionService getService() {
      return VoiceRecognitionService.this;
    }
  }

  @Override
  public IBinder onBind(Intent intent) {
    return binder;
  }

  @Override
  public void onCreate() {

    super.onCreate();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
      throw new RuntimeException("RECORD_AUDIO permission missing");
    }

    // Load the labels for the voice recognition model
    try {
      BufferedReader br =
          new BufferedReader(new InputStreamReader(getAssets().open(LABEL_FILENAME)));
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!", e);
    }

    // Set up an object to smooth recognition results to increase accuracy.
    recognizeCommands =
        new RecognizeCommands(
            labels,
            AVERAGE_WINDOW_DURATION_MS,
            DETECTION_THRESHOLD,
            SUPPRESSION_MS,
            MINIMUM_COUNT,
            MINIMUM_TIME_BETWEEN_SAMPLES_MS);

    try {
      tfLiteModel = loadModelFile(getAssets(), MODEL_FILENAME);
      recreateInterpreter();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    // enable Bluetooth headset (recommended for voice recognition due to disturbance by robot
    // motors
    if (getSharedPreferences(SharedPreferencesManager.PREFERENCES_NAME, Context.MODE_PRIVATE)
        .getBoolean(SharedPreferencesManager.BT_HEADSET, true)) {
      enableBTHeadset();
    }
  }

  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
    super.onStartCommand(intent, flags, startId);
    if (intent.getAction() != null) {
      switch (intent.getAction()) {
        case CMD_START_LISTEN:
          if (getSharedPreferences(SharedPreferencesManager.PREFERENCES_NAME, Context.MODE_PRIVATE)
              .getBoolean(SharedPreferencesManager.BT_HEADSET, true)) {
            enableBTHeadset();
          }
          startRecording();
          startRecognition();
          break;
        case CMD_STOP_LISTEN:
          stopRecognition();
          stopRecording();
          break;
      }
    } else { // possible on system recreate ?
      if (getSharedPreferences(SharedPreferencesManager.PREFERENCES_NAME, Context.MODE_PRIVATE)
          .getBoolean(SharedPreferencesManager.BT_HEADSET, true)) {
        enableBTHeadset();
      }
      startRecording();
      startRecognition();
    }
    return START_STICKY;
  }

  @Override
  public void onDestroy() {
    if (audioManager != null && btHeadsetConnected) {
      audioManager.stopBluetoothSco();
      // unregisterReceiver(null);
    }
    stopRecognition();
    stopRecording();
    super.onDestroy();
  }

  public synchronized void startRecording() {

    if (recordingThread != null) {
      return;
    }

    shouldContinue = true;
    recordingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                record();
              }
            });
    recordingThread.start();
  }

  public synchronized void stopRecording() {

    if (recordingThread == null) {
      return;
    }

    stopRecognition();
    shouldContinue = false;
    recordingThread = null;
  }

  /** Enable Bluetooth Headset for audio/voice commands */
  private void enableBTHeadset() {

    audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
    if (!audioManager.isBluetoothScoAvailableOffCall()) {
      Toast.makeText(
              getBaseContext(), "Bluetooth headset not supported on this phone", Toast.LENGTH_LONG)
          .show();
      audioManager = null;
      return;
    }

    if (!btHeadsetConnected) { // initally set callback for Bluetooth headset state
      registerReceiver(
          new BroadcastReceiver() {

            @Override
            public void onReceive(Context context, Intent intent) {
              int state = intent.getIntExtra(AudioManager.EXTRA_SCO_AUDIO_STATE, -1);
              Timber.d("Audio SCO state: " + state);
              if (AudioManager.SCO_AUDIO_STATE_CONNECTED == state) {
                btHeadsetConnected = true;
                Timber.d("SCO audio connected");
              } else if (state == AudioManager.SCO_AUDIO_STATE_DISCONNECTED) {
                btHeadsetConnected = false;
                Timber.d("SCO audio disconnected");
                unregisterReceiver(this);
              }
            }
          },
          new IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED));
    }
    Timber.i("Starting audio via bluetooth headset");
    audioManager.startBluetoothSco();
  }

  /** Called in own thread, continuous fills buffer with recorded audio */
  private void record() {
    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

    // Estimate the buffer size we'll need for this device.
    int bufferSize =
        AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = SAMPLE_RATE * 2;
    }
    short[] audioBuffer = new short[bufferSize / 2];

    AudioRecord record =
        new AudioRecord(
            AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            recordingBuffer.length);

    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      return;
    }

    record.startRecording();

    // Loop, gathering audio data and copying it to a round-robin buffer.
    while (shouldContinue) {

      int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
      int maxLength = recordingBuffer.length;
      int newRecordingOffset = recordingOffset + numberRead;
      int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
      int firstCopyLength = numberRead - secondCopyLength;
      // We store off all the data for the recognition thread to access. The ML
      // thread will copy out of this buffer into its own, while holding the
      // lock, so this should be thread safe.
      recordingBufferLock.lock();
      try {
        System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
        System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
        recordingOffset = newRecordingOffset % maxLength;
      } finally {
        recordingBufferLock.unlock();
      }
    }

    record.stop();
    record.release();
  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                recognize();
              }
            });
    recognitionThread.start();
  }

  public synchronized void stopRecognition() {
    if (recognitionThread == null) {
      return;
    }
    shouldContinueRecognition = false;
    recognitionThread = null;
  }

  /**
   * Callback to signal application on new recognized command
   *
   * @param dataCallback void proc(String)
   */
  public void setDataCallback(IDataReceived dataCallback) {
    this.dataReceivedCallback = dataCallback;
  }

  private void recognize() {

    short[] inputBuffer = new short[RECORDING_LENGTH];
    float[][] floatInputBuffer = new float[RECORDING_LENGTH][1];
    float[][] outputScores = new float[1][labels.size()];
    int[] sampleRateList = new int[] {SAMPLE_RATE};

    // Loop, grabbing recorded data and running the recognition model on it.
    while (shouldContinueRecognition) {

      // The recording thread places data in this round-robin buffer, so lock to
      // make sure there's no writing happening and then copy it to our own
      // local version.
      recordingBufferLock.lock();
      try {
        int maxLength = recordingBuffer.length;
        int firstCopyLength = maxLength - recordingOffset;
        int secondCopyLength = recordingOffset;
        System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
        System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
      } finally {
        recordingBufferLock.unlock();
      }

      int minbufval = 32767;
      int maxbufval = -32767;
      // We need to feed in float values between -1.0f and 1.0f, so divide the
      // signed 16-bit inputs.
      for (int i = 0; i < RECORDING_LENGTH; ++i) {
        floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f;
        if (inputBuffer[i] > maxbufval) maxbufval = inputBuffer[i];
        if (inputBuffer[i] < minbufval) minbufval = inputBuffer[i];
      }
      if (maxbufval - minbufval != 000) { // avoid recognition on low level noise
        Object[] inputArray = {floatInputBuffer, sampleRateList};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputScores);

        // Run the model.
        tfLiteLock.lock();
        try {
          tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        } finally {
          tfLiteLock.unlock();
        }

        // Use the smoother to figure out if we've had a real recognition event.
        long currentTime = System.currentTimeMillis();
        final RecognizeCommands.RecognitionResult result =
            recognizeCommands.processLatestResults(outputScores[0], currentTime);

        // If we do have a new command, send to bot control.
        if (result.isNewCommand && !result.foundCommand.startsWith("_")) {
          Timber.d("voice command found: " + result.foundCommand);
          dataReceivedCallback.dataReceived(result.foundCommand);
        }

        try {
          // We don't need to run too frequently, so snooze for a bit.
          Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
        } catch (InterruptedException e) {
          // Ignore
        }
      }
    }
  }

  private void recreateInterpreter() {
    tfLiteLock.lock();
    try {
      if (tfLite != null) {
        tfLite.close();
        tfLite = null;
      }
      tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
      tfLite.resizeInput(0, new int[] {RECORDING_LENGTH, 1});
      tfLite.resizeInput(1, new int[] {1});
    } finally {
      tfLiteLock.unlock();
    }
  }

  public boolean isListen() {
    return shouldContinue;
  }

  /**
   * Write recorded audio buffer to wav file,
   *
   * @param sampleRate sample frequency
   * @param sampleSize bit size of one value (short = 16)
   * @param channels mono =1 or stereo =2
   * @param duration recording length in sec
   * @param file output file
   * @param srcarray array to write to file
   * @throws IOException
   */
  public void createWaveFile(
      int sampleRate, short sampleSize, short channels, int duration, File file, short[] srcarray) {
    // example: createWaveFile(SAMPLE_RATE, (short) 16,(short) 1, SAMPLE_DURATION_MS / 1000, new
    // File(OpenBotApplication.getContext().getFilesDir(),result.foundCommand + ".wav"),
    // inputBuffer);
    // calculate some wav header values
    short blockAlign = (short) (sampleSize * channels / 8);
    int byteRate = sampleRate * sampleSize * channels / 8;
    int audioSize = byteRate * duration;
    int fileSize = audioSize + 44;

    // convert short to byte array
    /*   byte[] audioData = new byte[audioSize];
    for (int n=0; n < srcarray.length; n++) {
      byte lsb = (byte) (srcarray[n] & 0xff);
      byte msb = (byte) ((srcarray[n] >> 8) & 0xff);
        audioData[2*n]   = lsb;
        audioData[2*n+1] = msb;
    }    */
    ByteBuffer buffer = ByteBuffer.allocate(srcarray.length * 2);
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.asShortBuffer().put(srcarray);
    byte[] audioData = buffer.array();
    try {
      DataOutputStream out = new DataOutputStream(new FileOutputStream(file));
      // Write wav Header
      out.writeBytes("RIFF"); // 0-4 ChunkId always RIFF
      out.writeInt(
          Integer.reverseBytes(fileSize)); // 5-8 ChunkSize always audio-length +header-length(44)
      out.writeBytes("WAVE"); // 9-12 Format always WAVE
      out.writeBytes("fmt "); // 13-16 Subchunk1 ID always "fmt " with trailing whitespace
      out.writeInt(Integer.reverseBytes(16)); // 17-20 Subchunk1 Size always 16
      out.writeShort(Short.reverseBytes((short) 1)); // 21-22 Audio-Format 1 for PCM PulseAudio
      out.writeShort(Short.reverseBytes(channels)); // 23-24 Num-Channels 1 for mono, 2 for stereo
      out.writeInt(Integer.reverseBytes(sampleRate)); // 25-28 Sample-Rate
      out.writeInt(Integer.reverseBytes(byteRate)); // 29-32 Byte Rate
      out.writeShort(Short.reverseBytes(blockAlign)); // 33-34 Block Align
      out.writeShort(Short.reverseBytes(sampleSize)); // 35-36 Bits-Per-Sample
      out.writeBytes("data"); // 37-40 Subchunk2 ID always data
      out.writeInt(Integer.reverseBytes(audioSize)); // 41-44 Subchunk 2 Size audio-length

      out.write(audioData);
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
