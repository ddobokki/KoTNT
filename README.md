# TNT๐งจ Model (Text -> Num -> Text Model)

ํ๊ตญ์ด ๋ฌธ์ฅ์์ ์ซ์๊ฐ ๋ค์ด๊ฐ ๋ฌธ์ฅ์ ํ๊ธ๋ก ๋ณํ ํด์ฃผ๋ ๋ชจ๋ธ์๋๋ค.
๊ทธ ๋ฐ๋๋ ๊ฐ๋ฅํฉ๋๋ค.

์ซ์๋ฅผ ๊ฒฝ์ฐ์ ๋ฐ๋ผ ์ฝ๋ ๋ฐฉ๋ฒ์ด ๋ฌ๋ผ์ง๊ฒ ๋๋ฏ๋ก ์ด๋ฅผ ๋ฅ๋ฌ๋ ๋ชจ๋ธ์ ํ์ฉํด ํด๊ฒฐํ๋ ค๋ ํ ์ด ํ๋ก์ ํธ์๋๋ค.

ex)
- ๊ฒฝ์  1๋ฒ์ง <-> ๊ฒฝ์  ์ผ๋ฒ์ง
- ์ค๋ 1์์ ๋ณด๋๊ฒ ์ด๋? <-> ์ค๋ ํ ์์ ๋ณด๋๊ฒ ์ด๋?

# ์ฌ์ฉ ๋ฐ์ดํฐ
- ๋ณธ repo์์๋ ์์ฑ ๋ฐ์ดํฐ์ ์ ๋ต ์คํฌ๋ฆฝํธ๋ฅผ [KoSpeech](https://github.com/sooftware/kospeech) ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ฌ์ฉํ์ฌ trn ํ์ผ๋ก ์ ์ฒ๋ฆฌํ์ฌ ์ฌ์ฉํ์ต๋๋ค.
[KSponSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)๋ ์ด๊ณณ์์ ๋ค์ด ๋ฐ์ ์ ์์ต๋๋ค.
- ๊ณต๊ฐํ  ๋ชจ๋ธ์ ์ด์ธ์๋ [KoreanSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=130), [KconfSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=132), [KrespSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=87)์ ์คํฌ๋ฆฝํธ๋ฅผ ํ์ตํ ๋ชจ๋ธ์๋๋ค.

# ๋ฐ์ดํฐ ์ฒ๋ฆฌ
๋ง์ฝ AIHub์์ STT์ฉ ๋ฐ์ดํฐ๋ฅผ ๋ค์ด๋ฐ์ผ์จ๋ค๋ฉด, ์ด 3๋ฒ์ ๋ฐ์ดํฐ ์ฒ๋ฆฌ๊ฐ ํ์ํฉ๋๋ค.
1. (raw_transcripts_preproc) ๊ฐ Raw Transcripts๋ฅผ trnํ์ผ๋ก ๋ณ๊ฒฝํฉ๋๋ค.
2. (preprocess) trnํ์ผ์์ num sentence์ ko sentence์ ์ถ์ถํฉ๋๋ค. (์ฌ๊ธฐ์ 1์ฐจ csv ์ ์ฅ์ด ๋ฐ์ํฉ๋๋ค. out.csv)
3. (postprocess) ์ถ์ถํ ๋ฐ์ดํฐ๋ฅผ ํน์๋ฌธ์ ์ ์ฒ๋ฆฌ, ์ด์์น ๋ฐ์ดํฐ ์ ๊ฑฐ๋ค์ ์ฒ๋ฆฌ ํ csv๋ก ์ ์ฅํฉ๋๋ค. (์ฌ๊ธฐ์ 2์ฐจ csv ์ ์ฅ์ด ๋ฐ์ํฉ๋๋ค. post_out.csv)
## 1. raw_transcripts_preproc
### 1. Data Download
AIHub์์ ๋ฐ์ดํฐ๋ฅผ ๋ค์ด๋ก๋ ๋ฐ์ <br />
ํ๋ก์ ํธ์ /data/raw ํด๋์ ์์ถ ํ์ด์ ๋ฃ๊ธฐ

/data/raw/{์์ถํด์ ํ๋ฐ์ดํฐ๋ช}/config.yaml ์์ฑ <br />
ํด๋น yaml์ ์์ฑํ๋ ๋ฐฉ๋ฒ ๋ฐ ์์ค๋ ์์ผ๋ฉฐ, ์๋ ์ค๋ช์ ๋ฐ๋ผ ์ง์  ๊ตฌ์ฑํ์ฌ์ผ ํจ <br />
config.yml ์์ <br />
```yml
common:
  ๊ตฌ์ถ๊ธฐ๊ด: ์๋ฃจ๊ฒ์ดํธ
dir:
  train_flag_dir_dict: {True: "1.Training", False: "2.Validation"}
  parent_pattern: "[0-9].*"
  child_pattern: "**/[0-9]*"
dataset:
  domain_devset_cnt: {
    "1.๋ฐฉ์ก": 1,
    "2.์ทจ๋ฏธ": 1,
    "3.์ผ์์๋ถ": 1,
    "4.์ํ": 1,
    "5.๋ ์จ": 1,
    "6.๊ฒฝ์ ": 1,
    "7.๋์ด": 1,
    "8.์ผํ": 1,
  }
```
> ํด๋น ํ์ผ์ ๊ฐ์ ๊ตฌ์ถ๊ธฐ๊ด์ ๊ฐ์ ํด๋๊ตฌ์กฐ๋ฅผ ํ์ฉํ๋ค๋ ์ ์์ ์ฐฉ์ํ์ฌ, .trnํ์ผ์ ์ผ๊ด๋๊ฒ ๋ง๋ค๊ฒ ํ๊ธฐ ์ํจ <br />

common : ์ผ๋ฐ์ ์ธ data์ ํน์ง๋ค์ ๋ค๋ฃธ <br />
dir : ์๋ก ๋ค๋ฅธ ํด๋ ๊ท์น์ ์ ์ํ๊ธฐ ์ํด, ๊ฐ ๋ฐ์ดํฐ๋ณ๋ก ํน์ง์ ์์ฑํด์ค์ผํจ. <br />
 - train_flag_dir_dict : train/valid set ๊ตฌ์ฑ์ ํด๋๋ช <br />
 - parent,child_pattern : ์ค์  ๋ฉ์ธ, ์๋ธ๋๋ฉ์ธ์ ํด๋๋ช ์ ๊ท์ <br />

dataset : dev set ๋ฑ ๋ฐ์ดํฐ ์ถ๊ฐ ์ฒ๋ฆฌ์ ์ฌ์ฉํ  config <br />
### 2. Data Preprocess
```bash
bash shell_scripts/make_trn_for_make_script.sh
```
## 2. preprocess
```bash
python3 preprocess.py
```
## 3. (Optional) postprocess
```bash
python3 postprocess.py --csv_path={your_csv_dir}/{your_csv_filename}
```
ํน์ฌ๋ ์ถ๊ฐ์ ์ธ ์ ์ฒ๋ฆฌ ์๋ชป (์๋ฅผ๋ค๋ฉด feature, label์ด์ธ์ length 3 listํน์ 2๋ณด๋ค ์์ list) ๋๊ฑฐ๋ <br />
num๊ณผ ko ๋ฐ์ดํฐ๊ฐ ๊ธธ์ด๊ฐ ๋ค๋ฅธ๊ฒฝ์ฐ (๋ญ๊ฐ ๋งค์นญ์ด ์๋ชป๋๊ฑฐ๋ ์งค๋ฆฐ ๋ฐ์ดํฐ) <br />
์๋ง seq2seq์ ์ด์ฉํด ํ์ตํ๋ค๋ฉด, ๋ถํ์ํ ๋ฌธ์ฅ์ด ์์ธก๋์ด ์ ํ๋๊ฐ ๋ ๋จ์ด์ง ์๋ ์๋ค๊ณ  ํ๋จํ์ต๋๋ค. <br />
### ์์
![exam1](./readme_img/cut_exam1.jpg)
![exam2](./readme_img/cut_exam2.jpg)
์ ํ ์ด์ํ ๋ฌธ์ฅ์ฒ๋ผ ๋งค์นญ๋๋ ๊ฒฝ์ฐ๋ ์์

๊ฒฐ๋ก ์ ์ผ๋ก, ์ด๋ฐ ๋ฐ์ดํฐ๋ค์ ์ฒ๋ฆฌํ์ฌ post_out.csv๋ก ๋ง๋ค์ด์ค๋๋ค. <br />
์๋ฆฌ๋ sequencematcher ์ ์ฌ๋๋ฅผ ์ฌ์ฉํ๊ณ , ์ธ๊ฐ ํ๋จ์ ์๊ณ์น๋ฅผ ํ์ฉํ์ผ๋ฏ๋ก, 100% ๋ณด์ฅ์ ์๋์ง๋ง ์ ์ฒด ๋ฐ์ดํฐ์ 15%๋ฅผ ์ ์๊ฒ์ฌ ํ์ ์ ์ค์ฐจ๋ฒ์๋ ์์์ต๋๋ค.

# ๋ชจ๋ธ Repo (HuggingFace๐ค)
๋ชจ๋ธ๊ณผ ์ฑ๋ฅ ๋ฑ์ ํ์ธํ๋ ค๋ฉด ํ๊ธฐ URL์ ์ฐธ๊ณ ํ์ธ์. <br />
ko-barTNumText: https://huggingface.co/lIlBrother/ko-barTNumText <br />
ko-TextNumbarT: https://huggingface.co/lIlBrother/ko-TextNumbarT