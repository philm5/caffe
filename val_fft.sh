rm val_fft.txt
./build/tools/caffe test -model models/bvlc_reference_caffenet/test_fft.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0 -iterations 140 &> val_fft.txt
