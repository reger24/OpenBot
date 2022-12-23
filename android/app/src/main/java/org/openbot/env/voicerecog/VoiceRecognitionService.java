package org.openbot.env.voicerecog;

import android.Manifest;
import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Binder;
import android.os.IBinder;
import android.util.Log;
import androidx.core.app.ActivityCompat;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import org.tensorflow.lite.Interpreter;

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

    // Load the labels for the model, but only display those that don't start
    // with an underscore.
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
  }

  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
    if (intent.getAction() != null) {
      switch (intent.getAction()) {
        case CMD_START_LISTEN:
          startRecording();
          startRecognition();
          break;
        case CMD_STOP_LISTEN:
          stopRecognition();
          stopRecording();
          break;
      }
    }
    return START_STICKY;
  }

  @Override
  public void onDestroy() {
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

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
      // TODO: Consider calling
      //    ActivityCompat#requestPermissions
      // here to request the missing permissions, and then overriding
      //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
      //                                          int[] grantResults)
      // to handle the case where the user grants the permission. See the documentation
      // for ActivityCompat#requestPermissions for more details.
      return;
    }
    AudioRecord record =
        new AudioRecord(
            MediaRecorder.AudioSource.DEFAULT,
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
      if (maxbufval - minbufval > 2000) { // avoid recognition on low level noise
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
          // TODO: send found command to bot via a communication channel
          //  e.g.  ControllerToBotEventBus.emitEvent(commandStr);
          Log.i("voice command found: ", result.foundCommand);
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
}
