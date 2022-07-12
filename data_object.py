import json
import os.path
import subprocess
import time
import re
import pysrt
import pandas as pd
from pandas import DataFrame
from datetime import timedelta
from pympi.Elan import Eaf as Eaf_

punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'¬'"""
non_end = ('a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta',
           'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía', 'y', 'e', 'ni', 'na',
           'que', 'se', 'lles'
                        'o', 'u', 'ora', 'bien', 'tanto', 'como', 'cuanto', 'así', 'igual', 'mismo', 'sino', 'también',
           'pero',
           'mas', 'empre', 'mientras', 'sino', 'o', 'u', 'ya', 'ora', 'fuera', 'sea', 'porque', 'como', 'dado', 'visto',
           'puesto', 'pues', 'si', 'sin', 'aunque', 'aún', 'mientras', 'salvo', 'luego', 'conque', 'ergo', 'solo',
           'siempre', 'nunca', 'el', 'la', 'lo', 'los', 'las', 'un', 'una', 'uno', 'unas', 'unos', 'esto', 'esta',
           'estos', 'estas', 'aquello', 'aquella', 'aquellos', 'aquellas', 'eso', 'esa', 'esos', 'esas', 'mi', 'su',
           'tu', 'mis', 'sus', 'súas', 'seus', 'tus', 'mío', 'míos', 'tuyo', 'tuyos', 'suyo', 'suyos', 'son', 'unha',
           'fóra', 'sen', 'do', 'dos', 'da', 'das', 'debaixo', 'despois', 'estou', 'embaixo', 'ao', 'ademais', 'oposto',
           'máis', 'dende', 'ata', 'a', 'sen', 'dúas', 'lonxe', 'petro', 'ademais', 'adiante', 'aqueles')


class Converter(object):

    def __init__(self, path_in: str, ext_in: str, path_out: str, ext_out: str):
        transformation = {"json": self.google_json_to_ctm,
                          "stl": self.stl_to_srt,
                          "srt2stm": self.srt_to_stm,
                          "srt2txt": self.srt_to_plain_transcription,
                          "eaf2csv": self.eaf_to_csv,
                          "eaf2srt": self.eaf_to_srt
                          }

        trans = ext_in.lower()
        if trans not in transformation.keys():
            trans += f"2{ext_out.lower()}"

        if trans in transformation.keys():
            self.ans = self.run_method_in_folder(path_in, transformation[trans], ext_in, path_out)
        else:
            raise TypeError("Invalid transformation.")

    def __len__(self):
        return len(self.ans)

    @staticmethod
    def google_json_to_ctm(path_in_json: str, path_out_ctm: str, ctm_id: str = None):
        ctm_data = []
        with open(path_in_json, 'r') as f:
            data = f.read()

        json_dict = json.loads(data)

        for result in json_dict["results"]:
            if "alternatives" in result:
                alternatives = result["alternatives"][0]
                if "confidence" in alternatives:
                    confidence = alternatives["confidence"]
                else:
                    confidence = 0

                if not "words" in alternatives or len(alternatives) == 0:
                    pass
                else:
                    for words in alternatives["words"]:
                        start_time = float(words["startTime"][:-1])
                        end_time = float(words["endTime"][:-1])
                        word = words["word"]

                        start_time = start_time if (end_time - start_time < 0.8) else (end_time - 0.8)
                        duration = end_time - start_time

                        if not ctm_id:
                            ctm_id = path_in_json.replace(".json", "")

                        ctm_data.append(
                            " ".join([ctm_id, "SPK00", str(start_time), str(duration), word, str(confidence), '\n']))

        if not path_out_ctm:
            path_out_ctm = path_in_json.replace(".json", ".ctm")
        else:
            path_out_ctm = path_out_ctm.replace(".json", ".ctm")

        with open(path_out_ctm, 'w') as f:
            f.writelines(ctm_data)

        return ctm_data

    @staticmethod
    def stl_to_srt(path_in_stl: str, path_out_srt: str = None):
        path_in_stl = re.sub(" ", "_", path_in_stl)
        path_out_srt = re.sub(" ", "_", path_out_srt)

        os.renames(path_in_stl, path_in_stl)
        path_out_srt = path_out_srt.replace(".stl", ".srt")
        subprocess.Popen(
            ["tt", "convert", "-i", f'{path_in_stl}', "-o", f'{path_out_srt}'])

        with open(path_out_srt, 'r') as f:
            return f.readlines()

    @staticmethod
    def srt_to_stm(path_in_srt: str, path_out_stm: str = None):

        if not path_out_stm:
            path_out_stm = path_in_srt.replace(".srt", ".stm")
        else:
            path_out_stm = path_out_stm.replace(".srt", ".stm")

        subprocess.Popen(["convert_transcript", f'{path_in_srt}', f'{path_out_stm}'])

        with open(path_out_stm, 'r') as f:
            return f.readlines()

    @staticmethod
    def srt_to_plain_transcription(path_in_srt: str, path_out_txt: str = None):
        transcription = ""

        subtitle = pysrt.open(path_in_srt)
        for line in subtitle:
            text = line.text
            if "[" in text or "]" in text:
                pass
            else:
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\n', ' ', text)
                text = re.sub(r" +", " ", text)
                transcription += text + " "

        if not path_out_txt:
            path_out_txt = path_in_srt.replace(".srt", ".txt")
        elif ".srt" in path_out_txt:
            path_out_txt = path_out_txt.replace(".srt", ".txt")
        with open(path_out_txt, 'w') as f:
            f.write(transcription[:-1])

    @staticmethod
    def eaf_to_csv(path_in_eaf: str, path_out_csv: str = None):

        if not path_out_csv:
            path_out_csv = path_in_eaf.replace(".eaf", ".csv")
        else:
            path_out_csv = path_out_csv.replace(".eaf", ".csv")

        eaf = Eaf(path_in_eaf)
        csv = eaf.write_csv(path_out_csv)

        return csv

    @staticmethod
    def eaf_to_srt(path_in_eaf: str, path_out_srt: str = None):

        if not path_out_srt:
            path_out_srt = path_in_eaf.replace(".eaf", ".srt")
        else:
            path_out_srt = path_out_srt.replace(".eaf", ".srt")

        ans_srt = []
        df_speech = Eaf(path_in_eaf).df_speech
        for num_line in df_speech.iterrows():
            row_df = num_line[1]

            text = row_df.text_orto
            time_end = timedelta(seconds=row_df.time_end / 1000)
            time_init = timedelta(seconds=row_df.time_init / 1000)

            msg = f"{int(num_line[0]) + 1}\n{time_init} --> {time_end}\n{text}\n\n"
            ans_srt.append(msg)

        with open(path_out_srt, 'w') as f:
            f.writelines(ans_srt)

        return ans_srt

    @staticmethod
    def run_method_in_folder(folder_path: str, method, ext: str, output_path: str = None):
        list_of_ans = []
        for file in os.listdir(folder_path):
            if ext in file and os.path.isfile(folder_path + file):
                ans = method(folder_path + file, output_path + file)
                list_of_ans.append(ans)

        return list_of_ans

    def get_transformation(self):
        return self.ans


class SubtitleMaker(object):
    def __init__(self, input_data: str, path_out: str = "", extension: str = "json"):
        self.all_subtitles: list = []

        file_ext = input_data.split(".")[-1]
        if os.path.isfile(input_data) and file_ext == extension:
            print(f".srt DONE! ---> {input_data}")
            self.all_subtitles.append([Subtitle(input_data, path_out)])

        elif os.path.isdir(input_data):
            for file in os.listdir(input_data):
                file_ext = file.split(".")[-1]
                path_to_file = os.path.join(input_data, file)
                if file_ext == extension:
                    print(f".srt DONE! ---> {path_to_file}")
                    self.all_subtitles.append([Subtitle(path_to_file, path_out)])

    def get_all_subtitles(self):
        return self.all_subtitles


class Subtitle(object):
    def __init__(self, input_data, path_out: str = ""):
        self.file_path: str
        self.annotation: dict

        if isinstance(input_data, str):
            self.file_path = input_data
            self.annotation = self.input_to_annotation(input_data)

        elif isinstance(input_data, dict):
            self.file_path = path_out
            self.annotation = input_data
        else:
            raise TypeError("Unsupported type for annotation")

        self.subtitle = self.make_subtitle()
        self.write_srt(self.file_path)

    def input_to_annotation(self, input_data: str):
        ans = {}
        if os.path.isfile(input_data):
            if ".json" in input_data:
                ans = self.google_json_to_annotation(input_data)
            elif ".ctm" in input_data:
                ans = self.ctm_to_annotation(input_data)

        return ans

    @staticmethod
    def ctm_to_annotation(path_to_ctm):

        # Read .ctm
        with open(path_to_ctm, 'r', encoding="ISO-8859-1") as f:
            ctm_file = f.readlines()

        # Compose .ctm
        times_init, times_ends, words, confidences = [], [], [], []
        for line in ctm_file:
            line = line.strip().split()
            times_init.append(float(line[2]))
            times_ends.append(float(line[3]))
            words.append(line[4])
            confidences.append(float(line[5]))

        spk_ids: list = ['UNK'] * len(times_init)
        spk_confidences: list = [None] * len(times_init)
        acpr_confidences: list = spk_confidences.copy()

        # Make the annotation
        annotation: dict = {}
        for ind in range(len(times_init)):
            annotation[times_init[ind]] = [times_ends[ind], words[ind], confidences[ind],
                                           acpr_confidences[ind], spk_ids[ind], spk_confidences[ind]]

        return annotation

    @staticmethod
    def google_json_to_annotation(path_to_json):
        ans = {}
        with open(path_to_json, 'r') as f:
            data = f.read()
        json_dict = json.loads(data)
        for result in json_dict["results"]:
            if "alternatives" in result:
                alternatives = result["alternatives"][0]
                if "confidence" in alternatives:
                    confidence = alternatives["confidence"]
                else:
                    confidence = 0

                for words in alternatives["words"]:
                    start_time = float(words["startTime"][:-1])
                    end_time = float(words["endTime"][:-1])
                    word = words["word"]

                    start_time = start_time if (end_time - start_time < 0.8) else (end_time - 0.8)
                    duration = end_time - start_time
                    ans[start_time] = [duration, word, confidence, 0, "SPK00", 0]

        return ans

    @staticmethod
    def color_subtitle(uncolored_subtitle):
        def msec2srttime(msecs):
            secs, rmsecs = divmod(msecs, 1000)
            mins, secs = divmod(secs, 60)
            hours, mins = divmod(mins, 60)
            return '%02d:%02d:%02d,%03d' % (hours, mins, secs, rmsecs)

        def conf2color(srt_words, srt_conf, srt_conf_cap_punt):
            q = chr(34)
            yellow_font = '<font color=' + q + 'yellow' + q + '>'
            orange_font = '<font color=' + q + 'orange' + q + '>'
            red_font = '<font color=' + q + 'red' + q + '>'
            end_font = '</font>'

            def put_color(unc_word, x_conf):
                if x_conf > 0.9:
                    return unc_word
                elif x_conf > 0.7:
                    return yellow_font + unc_word + end_font
                elif x_conf > 0.5:
                    return orange_font + unc_word + end_font
                else:
                    return red_font + unc_word + end_font

            colered_line = ""
            for ind_word, word in enumerate(srt_words):
                if word.isalnum() and word == word.lower():  # just asr confident to lower case
                    colered_line += put_color(word, srt_conf[ind_word]) + " "
                elif word == word.lower() and not word.isalpha():
                    colered_line += put_color(word[:-1], srt_conf[ind_word])
                    colered_line += put_color(word[-1], srt_conf_cap_punt[ind_word]) + " "
                elif word[0] == word[0].upper() and word != word.upper():
                    colered_line += put_color(word[0], srt_conf_cap_punt[ind_word])
                    if word.isalnum():
                        colered_line += put_color(word[1:], srt_conf[ind_word]) + " "
                    else:
                        colered_line += put_color(word[1:-1], srt_conf[ind_word])
                        colered_line += put_color(word[-1], srt_conf_cap_punt[ind_word]) + " "
                else:
                    colered_line += put_color(word, min(srt_conf[ind_word], srt_conf_cap_punt[ind_word]))

            return colered_line

        colored_subtitle = []
        for ind, line in enumerate(uncolored_subtitle):
            colored_sub = conf2color(line[2], line[3], line[4])
            colored_sub = f'({line[-1]}) ' + colored_sub
            colored_subtitle.append([ind + 1, msec2srttime(line[0] * 1000), msec2srttime(line[1] * 1000), colored_sub])
        return colored_subtitle

    def make_lines_from_annotation(self):

        def split_1st_max_char_duration(annotation):
            lines, line = {1: ''}, []
            ind, line_duration, line_char_len = 0, 0, 0

            times = list(annotation.keys())
            start_time = times[0]
            start_spk = annotation[start_time][4]

            while ind < len(annotation):
                values = annotation[times[ind]]
                duration, word, spk = values[0], values[1], values[4]

                line_duration += duration
                line_char_len += len(word) + 1

                if ind == len(annotation) - 1 and spk == start_spk:
                    line.append(times[ind])
                    srt_indx = list(lines.keys())[-1]
                    lines[srt_indx] = line

                elif ind == len(annotation) - 1 and spk != start_spk:
                    srt_indx = list(lines.keys())[-1]
                    lines[srt_indx] = line
                    lines[srt_indx + 1] = [times[ind]]

                elif (line_duration <= 4 and line_char_len < 55) and spk == start_spk:
                    line.append(times[ind])

                    if word[-1] in punctuation:
                        start_time = times[ind + 1]
                        srt_indx = list(lines.keys())[-1]
                        lines[srt_indx] = line
                        lines[srt_indx + 1] = start_time

                        line = [start_time]
                        start_spk = annotation[start_time][4]
                        line_duration, line_char_len = 0, 0
                        ind += 1

                elif (line_duration >= 3 or line_char_len >= 55) or spk != start_spk:
                    start_time = times[ind]
                    if spk == start_spk and annotation[line[-1]][1] in non_end:
                        line.append(start_time)
                        start_time = times[ind + 1]
                        ind += 1

                    srt_indx = list(lines.keys())[-1]
                    lines[srt_indx] = line
                    lines[srt_indx + 1] = [start_time]

                    line = [start_time]
                    start_spk = annotation[start_time][4]
                    line_duration, line_char_len = 0, 0

                ind += 1

            return lines

        def split_2nd_build_lines(annotation, lines: dict):
            act_spk = annotation[lines[1][0]][4]
            ind, subtitles = 1, []

            while ind <= len(lines):
                words, asr_conf, punct_conf, spk_id = [], [], [], ''
                start_time_line = lines[ind][0]
                dur_last_word = annotation[lines[ind][-1]][0]
                end_time_line = lines[ind][-1] + dur_last_word

                for position, value in enumerate(lines[ind]):

                    asr_conf.append(annotation[value][2])
                    punct_conf.append(annotation[value][3])
                    spk_id = annotation[value][4]

                    word = annotation[value][1]
                    if position == 0 and spk_id != act_spk:
                        word = word[0].upper() + word[1:]
                        act_spk = spk_id
                    words.append(word)

                subtitles.append([start_time_line, end_time_line, words, asr_conf, punct_conf, spk_id])
                ind += 1

            return subtitles

        def split_3rd_refine(raw_subtitle: list):
            def sum_lines(line_1: list, line_2: list):
                line_1[1] = line_2[1]
                line_1[2].extend(line_2[2])
                line_1[3].extend(line_2[3])
                line_1[4].extend(line_2[4])

                return line_1

            ind = 0
            final_subtitle = []
            while ind < len(raw_subtitle):
                spk_id = raw_subtitle[ind][-1]
                if ind < len(raw_subtitle) - 1 and spk_id == raw_subtitle[ind + 1][-1] and len(
                        raw_subtitle[ind + 1][2]) <= 2:
                    new_line = sum_lines(raw_subtitle[ind], raw_subtitle[ind + 1])
                    final_subtitle.append(new_line)
                    ind += 2
                else:
                    final_subtitle.append(raw_subtitle[ind])
                    ind += 1

            return final_subtitle

        input_annotation = self.annotation.copy()
        raw_lines = split_1st_max_char_duration(input_annotation)
        raw_subtitles = split_2nd_build_lines(input_annotation, raw_lines)
        subtitle = split_3rd_refine(raw_subtitles)
        return subtitle

    def make_subtitle(self):
        subtitle = self.make_lines_from_annotation()
        colored_lines = self.color_subtitle(subtitle)
        return colored_lines

    def write_srt(self, path_out):
        if path_out == "":
            path_out = f"data/file_test_{time.time()}"
        else:
            file_extention = path_out.split(".")[-1]
            path_out = path_out.replace(file_extention, 'srt')

        with open(path_out, 'w') as f:
            for line in self.subtitle:
                f.write("\n%d\n%s --> %s\n%s\n" % (line[0], line[1], line[2], line[3]))


class Eaf(object):
    def __init__(self, path):
        self.path = path
        self.eaf = Eaf_(path)

        self.transcription, self.df_acoustic_events, self.df_speech = self.make_csv_df()

    @staticmethod
    def make_data_frame_from_tier(data_tier: list, df_in: pd.DataFrame = pd.DataFrame(), column: int = 2,
                                  fill_none_with: str = None, mach_tolerance: int = 350):
        df_out = df_in.copy()
        for i in data_tier:
            time_start = int(i[0])
            time_final = int(i[1])
            value = i[2]

            if df_in.empty:
                entry = pd.Series([time_start, time_final, None, None, None, None, None, None])
                entry[column] = value
                df_out = df_out.append(entry, ignore_index=True)
            else:
                index = df_out.loc[(df_out['time_init'] == time_start) | (df_out['time_end'] == time_final)].index
                if index.empty:
                    mach = df_out.loc[
                        (round(df_out['time_init'] / mach_tolerance) == round(time_start / mach_tolerance)) |
                        (round(df_out['time_end'] / mach_tolerance) == round(time_final / mach_tolerance))]
                    index = mach.index

                df_out.iloc[index, column] = value

        if fill_none_with:
            df_out.iloc[:, column] = df_out.iloc[:, column].fillna(fill_none_with)

        df_out.columns = ["time_init", "time_end", "text", "text_orto", "speaker", "language", "topic", "acoustic_info"]
        return df_out

    @staticmethod
    def norm_annotation(df_speech):
        all_lines = ""
        texts = df_speech["text"].copy()
        for ind, line in enumerate(texts):
            norm_line = ""
            line = line.split()

            for word in line:
                if "/" in word:
                    continue
                word = word.lower()
                word = re.sub(r'[^\w\s]', '', word)
                norm_line += word + " "

            texts[ind] = norm_line[:-1]
            all_lines += norm_line
        return all_lines, texts

    def make_csv_df(self):
        data = {}
        tiers_name = self.eaf.get_tier_names()
        for t in tiers_name:
            data[t] = self.eaf.get_annotation_data_for_tier(t)
        df_speech = self.make_data_frame_from_tier(data['Segment'])
        df_speech = self.make_data_frame_from_tier(data["Speakers"], df_speech, column=4, fill_none_with="UNK")
        df_speech = self.make_data_frame_from_tier(data["Language"], df_speech, column=5, fill_none_with="galego")
        df_speech = self.make_data_frame_from_tier(data["Topic"], df_speech, column=6)
        df_acoustic_events = self.make_data_frame_from_tier(data["Others"], column=7)

        all_lines, texts = self.norm_annotation(df_speech)

        df_speech: DataFrame = df_speech.assign(text_orto=texts)
        return all_lines, df_acoustic_events, df_speech

    def write_csv(self, path_csv: str = None):

        if not path_csv:
            path_csv = self.path.replace(".eaf", ".csv")

        self.df_speech.to_csv(path_csv)
        self.df_acoustic_events.to_csv(path_csv.replace(".csv", "_acoustic.csv"))
        return self.df_speech

    def write_srt(self, path_srt: str = None):

        ans_srt = []
        for num_line in self.df_speech.iterrows():
            row_df = num_line[1]

            spk = row_df.speaker
            text = row_df.text_orto
            time_end = timedelta(seconds=row_df.time_end / 1000)
            time_init = timedelta(seconds=row_df.time_init / 1000)

            msg = f"{num_line[0] + 1}\n{time_init} --> {time_end}\n[{spk}] {text}\n\n"
            ans_srt.append(msg)

        if not path_srt:
            path_srt = self.path.replace(".eaf", ".csv")

        with open(path_srt, 'w') as f:
            f.writelines(ans_srt)

        return ans_srt

    def write_transcription(self, path_trans: str = None):

        if not path_trans:
            path_trans = self.path.replace(".eaf", ".csv")

        with open(path_trans, "w") as f:
            f.write(self.transcription)


class WER(object):
    def __init__(self, path_to_binary: str, path_true: str, path_hipot: str, path_out: str = None):
        self.wer: str = ""

        self.path_true = path_true
        self.path_hip = path_hipot
        self.path_to_binary = path_to_binary
        self.path_out = path_out

    def wer_ctrl(self):
        ## TODO Implement this controller for run WER using folder of trues and hypothesis

        if os.path.isfile(self.path_true):
            path_true = [self.path_true]
        if os.path.isfile(self.path_hip):
            path_hip = [self.path_hip]

        # map_folder = {}
        # for true, hip in zip(path_true, path_hip):
        #     base_true

    def calculate_wer(self):
        id_wer = self.path_hip.split("/")[-1].split(".")[0]
        subprocess.Popen([self.path_to_binary, id_wer, f'{self.path_hip}', f'{self.path_true}'])

        path_wer = self.path_hip.replace(".ctm", "_wer")
        with open(path_wer, 'r') as f:
            self.wer = f.readlines()

        if self.path_out:
            os.remove(path_wer)
            with open(f'{self.path_out + id_wer}_wer.txt', 'w') as f:
                f.writelines(self.wer[1:])

    def get_wer(self):
        return self.wer
