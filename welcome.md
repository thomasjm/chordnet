---
comments:
- email: bayanbatn@gmail.com
  ts: 1537756764549
  content: commenting is cool

---


# Okay here's the new thing


```{python3}
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('rendered/C_maj.wav')
frequencies, times, spectrogram = signal.spectrogram(
  samples.mean(axis=-1), # stereo?
  sample_rate,
  window=signal.get_window('hann',2000),
  noverlap=1000,
  nperseg=2000,
)
print(len(frequencies))
```
<div class="python3">
    <div class="stdout">
        <pre>1001
</pre>
    </div>
</div>


```{python3}
plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(top=1200)
plt.xlim(right=0.5)
plt.show()
```
<div class="python3">
    <div class="display">
        <div class="display-data result-image-container" data-mime-type="image/png">
            <img class="result-image" src="assets/c770abe0d0.png" title="Result image" alt>
        </div>
    </div>
</div>



```{python3}
chord_variants = [
  '_maj', '_min', 'b_maj', 'b_min']
chord_classes = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G']
import itertools
chords = itertools.product(chord_classes, chord_variants)
chords = [row[0]+row[1] for row in chords]
label_to_chord = dict(enumerate(chords))
chord_to_label = dict((row[1], row[0]) for row in enumerate(chords))
```


# Generate Training Set

```{python3}
train_data = []
train_labels = []
for chord in chords:
  try:
    sample_rate, samples = wavfile.read(f'rendered/{chord}.wav')
  except FileNotFoundError:
    continue
  frequencies, times, spectrogram = signal.spectrogram(
    samples.mean(axis=-1), # stereo?
    sample_rate,
    window=signal.get_window('hann',2000),
    noverlap=1000,
    nperseg=2000,
  )
  train_data.append(spectrogram)
  train_labels.append(chord_to_label[chord])
  print(f'finished running for {chord} chord')
```
<div class="python3">
    <div class="stdout">
        <pre>finished running for A_maj chord
finished running for A_min chord
finished running for Ab_maj chord
finished running for Ab_min chord
finished running for B_maj chord
finished running for B_min chord
finished running for Bb_maj chord
finished running for Bb_min chord
finished running for C_maj chord
finished running for C_min chord
finished running for D_maj chord
finished running for D_min chord
finished running for Db_maj chord
finished running for Db_min chord
finished running for E_maj chord
finished running for E_min chord
finished running for Eb_maj chord
finished running for Eb_min chord
finished running for F_maj chord
finished running for F_min chord
finished running for G_maj chord
finished running for G_min chord
finished running for Gb_maj chord
finished running for Gb_min chord
</pre>
    </div>
</div>

```{python3}
print(train_data[0].shape)
print(train_data[1].shape)
print(len(train_data))
```
<div class="python3">
    <div class="stdout">
        <pre>(1001, 131)
(1001, 131)
24
</pre>
    </div>
</div>

```{python3}
frequencies
```
<div class="python3">
    <div class="result">
        <pre>array([    0.  ,    22.05,    44.1 , ..., 22005.9 , 22027.95, 22050.  ])</pre>
    </div>
</div>


# Train Trivial Model

```{bash}
pip3 install --user --upgrade tensorflow
```
<div class="bash">
    <div class="stdout">
        <pre>Collecting tensorflow
  Using cached https://files.pythonhosted.org/packages/04/7e/a484776c73b1431f2b077e13801531e966113492552194fe721e6ef88d5d/tensorflow-1.10.1-cp36-cp36m-manylinux1_x86_64.whl
Collecting setuptools&lt;=39.1.0 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/8c/10/79282747f9169f21c053c562a0baa21815a8c7879be97abd930dbcf862e8/setuptools-39.1.0-py2.py3-none-any.whl
Collecting tensorboard&lt;1.11.0,&gt;=1.10.0 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/c6/17/ecd918a004f297955c30b4fffbea100b1606c225dbf0443264012773c3ff/tensorboard-1.10.0-py3-none-any.whl
Collecting absl-py&gt;=0.1.6 (from tensorflow)
Collecting numpy&lt;=1.14.5,&gt;=1.13.3 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/68/1e/116ad560de97694e2d0c1843a7a0075cc9f49e922454d32f49a80eb6f1f2/numpy-1.14.5-cp36-cp36m-manylinux1_x86_64.whl
Collecting termcolor&gt;=1.1.0 (from tensorflow)
Collecting grpcio&gt;=1.8.6 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/a7/9c/523fec4e50cd4de5effeade9fab6c1da32e7e1d72372e8e514274ffb6509/grpcio-1.15.0-cp36-cp36m-manylinux1_x86_64.whl
Collecting astor&gt;=0.6.0 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/35/6b/11530768cac581a12952a2aad00e1526b89d242d0b9f59534ef6e6a1752f/astor-0.7.1-py2.py3-none-any.whl
Collecting gast&gt;=0.2.0 (from tensorflow)
Collecting wheel&gt;=0.26 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/81/30/e935244ca6165187ae8be876b6316ae201b71485538ffac1d718843025a9/wheel-0.31.1-py2.py3-none-any.whl
Collecting protobuf&gt;=3.6.0 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/c2/f9/28787754923612ca9bfdffc588daa05580ed70698add063a5629d1a4209d/protobuf-3.6.1-cp36-cp36m-manylinux1_x86_64.whl
Collecting six&gt;=1.10.0 (from tensorflow)
  Using cached https://files.pythonhosted.org/packages/67/4b/141a581104b1f6397bfa78ac9d43d8ad29a7ca43ea90a2d863fe3056e86a/six-1.11.0-py2.py3-none-any.whl
Collecting markdown&gt;=2.6.8 (from tensorboard&lt;1.11.0,&gt;=1.10.0-&gt;tensorflow)
  Using cached https://files.pythonhosted.org/packages/7a/fd/e22357c299e93c0bc11ec8ba54e79f98dd568e09adfe9b39d6852c744938/Markdown-3.0-py2.py3-none-any.whl
Collecting werkzeug&gt;=0.11.10 (from tensorboard&lt;1.11.0,&gt;=1.10.0-&gt;tensorflow)
  Using cached https://files.pythonhosted.org/packages/20/c4/12e3e56473e52375aa29c4764e70d1b8f3efa6682bef8d0aae04fe335243/Werkzeug-0.14.1-py2.py3-none-any.whl
Installing collected packages: setuptools, numpy, six, protobuf, markdown, werkzeug, wheel, tensorboard, absl-py, termcolor, grpcio, astor, gast, tensorflow
Successfully installed absl-py-0.5.0 astor-0.7.1 gast-0.2.0 grpcio-1.15.0 markdown-3.0 numpy-1.14.5 protobuf-3.6.1 setuptools-39.1.0 six-1.11.0 tensorboard-1.10.0 tensorflow-1.10.1 termcolor-1.1.0 werkzeug-0.14.1 wheel-0.31.1
</pre>
    </div>
</div>

```{bash}
python3 -c "import tensorflow as tf; print(tf.__version__)"
```
<div class="bash">
    <div class="stdout">
        <pre>1.10.1
</pre>
    </div>
</div>

```{bash}
pip3 install --user keras
```
<div class="bash">
    <div class="stdout">
        <pre>Collecting keras
[?25l  Downloading https://files.pythonhosted.org/packages/34/7d/b1dedde8af99bd82f20ed7e9697aac0597de3049b1f786aa2aac3b9bd4da/Keras-2.2.2-py2.py3-none-any.whl (299kB)

[?25hRequirement already satisfied: numpy&gt;=1.9.1 in /usr/local/lib/python2.7/dist-packages (from keras) (1.15.1)
Collecting keras-preprocessing==1.0.2 (from keras)
  Downloading https://files.pythonhosted.org/packages/71/26/1e778ebd737032749824d5cba7dbd3b0cf9234b87ab5ec79f5f0403ca7e9/Keras_Preprocessing-1.0.2-py2.py3-none-any.whl
Requirement already satisfied: scipy&gt;=0.14 in /usr/local/lib/python2.7/dist-packages (from keras) (1.1.0)
Collecting h5py (from keras)
[?25l  Downloading https://files.pythonhosted.org/packages/33/0c/1c5dfa85e05052aa5f50969d87c67a2128dc39a6f8ce459a503717e56bd0/h5py-2.8.0-cp27-cp27mu-manylinux1_x86_64.whl (2.7MB)

[?25hCollecting keras-applications==1.0.4 (from keras)
[?25l  Downloading https://files.pythonhosted.org/packages/54/90/8f327deaa37a71caddb59b7b4aaa9d4b3e90c0e76f8c2d1572005278ddc5/Keras_Applications-1.0.4-py2.py3-none-any.whl (43kB)

[?25hCollecting pyyaml (from keras)
[?25l  Downloading https://files.pythonhosted.org/packages/9e/a3/1d13970c3f36777c583f136c136f804d70f500168edc1edea6daa7200769/PyYAML-3.13.tar.gz (270kB)

[?25hRequirement already satisfied: six&gt;=1.9.0 in /usr/local/lib/python2.7/dist-packages (from keras) (1.11.0)
Building wheels for collected packages: pyyaml
  Running setup.py bdist_wheel for pyyaml ... [?25l- done
[?25h  Stored in directory: /home/user/.cache/pip/wheels/ad/da/0c/74eb680767247273e2cf2723482cb9c924fe70af57c334513f
Successfully built pyyaml
Installing collected packages: keras-preprocessing, h5py, keras-applications, pyyaml, keras
Successfully installed h5py-2.8.0 keras-2.2.2 keras-applications-1.0.4 keras-preprocessing-1.0.2 pyyaml-3.13
</pre>
    </div>
</div>



```{python3}
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import re
import pickle

from keras import backend as K
import keras.layers as layers
from keras.models import Model
```
<div class="python3">
    <div class="error" ename="ModuleNotFoundError" evalue="No module named 'tensorflow'">
        <pre class="traceback"><code><span class="ansi-red">---------------------------------------------------------------------------</span>
<span class="ansi-red">ModuleNotFoundError</span>                       Traceback (most recent call last)
<span class="ansi-green">&lt;ipython-input-1-7abd8b4d2699&gt;</span> in <span class="ansi-cyan">&lt;module&gt;</span><span class="ansi-blue">()</span>
<span class="ansi-green">----&gt; 1</span><span class="ansi-red"> </span><span class="ansi-green">import</span> tensorflow <span class="ansi-green">as</span> tf<span class="ansi-blue"></span>
<span class="ansi-bold"><span class="ansi-green">      2</span></span> <span class="ansi-green">import</span> numpy <span class="ansi-green">as</span> np<span class="ansi-blue"></span>
<span class="ansi-bold"><span class="ansi-green">      3</span></span> <span class="ansi-green">import</span> os<span class="ansi-blue"></span>
<span class="ansi-bold"><span class="ansi-green">      4</span></span> <span class="ansi-green">import</span> pandas <span class="ansi-green">as</span> pd<span class="ansi-blue"></span>
<span class="ansi-bold"><span class="ansi-green">      5</span></span> <span class="ansi-green">import</span> re<span class="ansi-blue"></span>

<span class="ansi-red">ModuleNotFoundError</span>: No module named 'tensorflow'</code></pre>
    </div>
</div>