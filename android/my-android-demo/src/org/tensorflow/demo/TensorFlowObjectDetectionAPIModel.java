/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TensorFlowObjectDetectionAPIModel implements Classifier {

  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int MAX_RESULTS = 100;

  // Config values.
  private String[] inputNames;
  private String[] outputNames;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intValues;

  // SSD-MobileNet-v1: input arrays to feed
  private Vector<String> labels = new Vector<>();
  private byte[] byteValues;

  // SSD-MobileNet-v1: output arrays from the SSD-MobileNet-v1
  private float[] outputLocations;
  private float[] outputScores;
  private float[] outputClasses;
  private float[] outputNumDetections;

//  // Mobile-Mask-RCNN: input arrays to feed
//  private float[] floatValues;
//  private float[] image_metas;
//  private float[] anchors;

//  // Mobile-Mask-RCNN: for reading the pre-calculated values
//  private Vector<String> v_anchors = new Vector<>();
//  private Vector<String> v_image_metas = new Vector<>();

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize) throws IOException {

    final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();
    d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
    d.inputSize = inputSize;
    final Graph g = d.inferenceInterface.graph();
    String line;

    // Pre-allocate buffers for reading teh camera
    d.intValues = new int[d.inputSize * d.inputSize];

    // SSD-MobileNet-v1: Input and output tensor
    d.inputNames = new String[] {"image_tensor"};
    d.outputNames = new String[] {"detection_boxes", "detection_scores", "detection_classes", "num_detections"};

    // SSD-MobileNet-v1: Pre-allocate buffers for input and output tensors
    d.byteValues = new byte[d.inputSize * d.inputSize * 3];
    d.outputScores = new float[MAX_RESULTS];
    d.outputLocations = new float[MAX_RESULTS * 4];
    d.outputClasses = new float[MAX_RESULTS];
    d.outputNumDetections = new float[1];

    // SSD-MobileNet-v1: Read the list of labels from file
    InputStream labelsInput;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

//    // Mobile-Mask-RCNN: Input and output tensor
//    d.inputNames = new String[] {"input_image", "input_image_matas", "input_anchors"};
//    d.outputNames = new String[] {"output_detections", "output_mrcnn_mask"};

//    // Mobile-Mask-RCNN: Pre-allocate buffers for input and output tensors
//    d.floatValues = new float[d.inputSize * d.inputSize * 3];
//    d.image_metas = new float[93];
//    d.anchors = new float[90660];
//    for(int i = 0; i < d.image_metas.length; ++i) {
//      d.image_metas[i] = Float.parseFloat(d.v_image_metas.get(i));
//    }
//    for(int i = 0; i < d.anchors.length; ++i) {
//      d.anchors[i] = Float.parseFloat(d.v_anchors.get(i));
//    }

//    // Mobile-Mask-RCNN: Read the 'anchors' and 'image_metas' from file as the input to feed the model
//    InputStream anchorsInput = assetManager.open("anchors.log");
//    InputStream metasInput = assetManager.open("image_metas.log");
//    BufferedReader br_a = new BufferedReader(new InputStreamReader(anchorsInput));
//    BufferedReader br_m = new BufferedReader(new InputStreamReader(metasInput));
//    while ((line = br_a.readLine()) != null) {
//      d.v_anchors.add(line);
//    }
//    while ((line = br_m.readLine()) != null) {
//      d.v_image_metas.add(line);
//    }
//    br_a.close();
//    br_m.close();

    // check whether the input and output tensors exist
    for(int k = 0; k < d.inputNames.length; ++k) {
      final Operation inputOp = g.operation(d.inputNames[k]);
      if (inputOp == null) {
        throw new RuntimeException("Failed to find input Node '" + d.inputNames[k] + "'");
      }
    }
    for(int k = 0; k < d.outputNames.length; ++k) {
      final Operation outputOp = g.operation(d.outputNames[k]);
      if (outputOp == null) {
        throw new RuntimeException("Failed to find output Node '" + d.outputNames[k] + "'");
      }
    }

    return d;
  }

  private TensorFlowObjectDetectionAPIModel() {}

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognize");

    Trace.beginSection("pre-process");

    // Preprocess the image data to extract R, G and B bytes from int of form 0x00RRGGBB on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    // pre-processing of the SSD-MobileNet-v1
    for (int i = 0; i < intValues.length; ++i) {
      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
      byteValues[i * 3] = (byte) ((intValues[i] >> 16) & 0xFF);
    }

//    // Mobile-Mask-RCNN: pre-processing
//    for (int i = 0; i < intValues.length; ++i) {
//      floatValues[i * 3 + 2] = (float)(intValues[i] & 0xFF) / (float)127.5 - (float)1.0;
//      floatValues[i * 3 + 1] = (float)((intValues[i] >> 8) & 0xFF) / (float)127.5 - (float)1.0;
//      floatValues[i * 3] = (float)((intValues[i] >> 16) & 0xFF) / (float)127.5 - (float)1.0;
//    }

    Trace.endSection();

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");

    // SSD-MobileNet-v1: feeding
    inferenceInterface.feed(inputNames[0], byteValues, 1, inputSize, inputSize, 3);

//    // Mobile-Mask-RCNN: feeding
//    inferenceInterface.feed(inputNames[0], floatValues, 1, inputSize, inputSize, 3);
//    inferenceInterface.feed(inputNames[1], image_metas, 1, 93);
//    inferenceInterface.feed(inputNames[2], anchors, 1, 25575, 4);

    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    outputLocations = new float[MAX_RESULTS * 4];
    outputScores = new float[MAX_RESULTS];
    outputClasses = new float[MAX_RESULTS];
    outputNumDetections = new float[1];
    inferenceInterface.fetch(outputNames[0], outputLocations);
    inferenceInterface.fetch(outputNames[1], outputScores);
    inferenceInterface.fetch(outputNames[2], outputClasses);
    inferenceInterface.fetch(outputNames[3], outputNumDetections);
    Trace.endSection();

    // Find the best detections.
    final PriorityQueue<Recognition> pq =
      new PriorityQueue<>(
        1,
        new Comparator<Recognition>() {
          @Override
          public int compare(final Recognition lhs, final Recognition rhs) {
            // Intentionally reversed to put high confidence at the head of the queue.
            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
          }
        });

    // Scale them back to the input size.
    for (int i = 0; i < outputScores.length; ++i) {
      final RectF detection =
        new RectF(
            outputLocations[4 * i + 1] * inputSize,
            outputLocations[4 * i] * inputSize,
            outputLocations[4 * i + 3] * inputSize,
            outputLocations[4 * i + 2] * inputSize);
      pq.add(
        new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
    }

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;

//    // Mobile-Mask-RCNN: do not assign a return
//    return new ArrayList<>();
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
