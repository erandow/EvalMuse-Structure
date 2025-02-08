# EvalMuse-Structure
This repo is prepared for EvalMuse Part2-Structure Distortion Detection.
Baseline for ntire 2025 'Text to Image Generation Model Quality Assessment -Track2- Structure Distortion Detection' has been released.

# Framework
This baseline (EM-RAHF) is inspired by paper [Rich Human Feedback for Text-to-Image Generation](https://arxiv.org/pdf/2312.10240), since the authors of RAHF did not provide code, we modified some details of the original model and achieved better performance.The details of our methods will be published in a technical report paper.
![baseline framework]()


# Baseline Results
We provide a Google Drive link for the baseline prediction result: [baseline result](https://drive.google.com/file/d/18OBQVFlpY6rr9EZapVKGWIJtaDyQ-_BQ/view?usp=drive_link)  
The metrics score of the baseline is:

| Precision      | Recall      | F1-score      | PLCC      | SROCC      |Final-score      |
|--------------|--------------|--------------|--------------|--------------|--------------|
| 0.5086   | 0.6728   | 0.5793   | 0.6945   | 0.6677   |0.6098  |

# Citation and Acknowledgement

If you find EvalMuse useful for your research, please consider cite our paper:
```bibtex
@misc{han2024evalmuse40kreliablefinegrainedbenchmark,
      title={EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation}, 
      author={Shuhao Han and Haotian Fan and Jiachen Fu and Liang Li and Tao Li and Junhui Cui and Yunqiu Wang and Yang Tai and Jingwei Sun and Chunle Guo and Chongyi Li},
      year={2024},
      eprint={2412.18150},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.18150}, 
}
```
For using baseline RAHF, please cite cite paper
