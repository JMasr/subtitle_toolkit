from data_object import *

DATA_TEST: str = "/home/jsanhcez/Documentos/Cursos/TensorBoard/subtitle_toolkit/data/data_test/"
JSON_FOLDER = DATA_TEST + "json/"
CTM_FOLDER = DATA_TEST + "ctm/"
STL_FOLDER = DATA_TEST + "stl/"
SRT_FOLDER = DATA_TEST + "srt/"
STM_FOLDER = DATA_TEST + "stm/"
TXT_FOLDER = DATA_TEST + "txt/"
EAF_FOLDER = DATA_TEST + "eaf/"
CSV_FOLDER = DATA_TEST + "csv/"

ctm = Converter(JSON_FOLDER, "json", CTM_FOLDER, "ctm")
srt = Converter(STL_FOLDER, "stl", SRT_FOLDER, "srt")
stm = Converter(SRT_FOLDER, "srt", STM_FOLDER, "stm")
txt = Converter(SRT_FOLDER, "srt", TXT_FOLDER, "txt")
eaf = Eaf(EAF_FOLDER + "test.eaf")
csv = eaf.write_csv(CSV_FOLDER + "test.csv")
print("DONE!")
