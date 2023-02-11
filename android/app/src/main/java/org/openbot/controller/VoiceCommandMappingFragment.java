package org.openbot.controller;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import org.jetbrains.annotations.NotNull;
import org.openbot.R;
import org.openbot.common.ControlsFragment;
import org.openbot.databinding.FragmentVoicecommandMappingBinding;

public class VoiceCommandMappingFragment extends ControlsFragment {

  private FragmentVoicecommandMappingBinding binding;
  private Handler handler;
  private HandlerThread handlerThread;

  @Override
  public void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
  }

  @Override
  public View onCreateView(
      @NotNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
    // Inflate the layout for this fragment
    binding = FragmentVoicecommandMappingBinding.inflate(inflater, container, false);
    return binding.getRoot();
  }

  @Override
  public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
    super.onViewCreated(view, savedInstanceState);
    if (isBluetoothHeadsetConnected()) {
      binding.headsetToggle.setVisibility(View.VISIBLE);
    }
  }

  @Override
  public synchronized void onResume() {
    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
    super.onResume();
  }

  @Override
  public synchronized void onPause() {
    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      e.printStackTrace();
    }
    super.onPause();
  }

  @Override
  protected void processControllerKeyData(String command) {}

  @Override
  protected void processUSBData(String data) {}

  /**
   * Pre-process voice recognition commands, highlight recognized command
   *
   * @param voicecommand received from vrservice
   */
  @Override
  protected void processVoiceRecognitionCommand(String voicecommand) {
    TextView selectedTextView;
    switch (voicecommand) {
      case "yes":
        selectedTextView = binding.yes;
        break;
      case "no":
        selectedTextView = binding.no;
        break;
      case "up":
        selectedTextView = binding.up;
        break;
      case "down":
        selectedTextView = binding.down;
        break;
      case "left":
        selectedTextView = binding.left;
        break;
      case "right":
        selectedTextView = binding.right;
        break;
      case "on":
        selectedTextView = binding.on;
        break;
      case "off":
        selectedTextView = binding.off;
        break;
      case "stop":
        selectedTextView = binding.stop;
        break;
      case "go":
        selectedTextView = binding.go;
        break;
      default:
        selectedTextView = null;
        break;
    }

    // highlight recognized text view
    if (selectedTextView != null) {
      selectedTextView.setBackgroundColor(getResources().getColor(R.color.colorPrimaryDark));
      selectedTextView.setTextColor(getResources().getColor(android.R.color.holo_orange_light));
      handler.postDelayed(
          new Runnable() {
            @Override
            public void run() {
              selectedTextView.setBackgroundColor(getResources().getColor(R.color.background));
              selectedTextView.setTextColor(getResources().getColor(R.color.colorPrimaryDark));
            }
          },
          750);
    }
  }
}
