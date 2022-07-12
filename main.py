from data_object import *

DATA_TEST: str = "./data/data_test"
JSON_FOLDER = DATA_TEST + "json/"
CTM_FOLDER = DATA_TEST + "ctm/"
STL_FOLDER = DATA_TEST + "stl/"
SRT_FOLDER = DATA_TEST + "srt/"
STM_FOLDER = DATA_TEST + "stm/"
TXT_FOLDER = DATA_TEST + "txt/"
EAF_FOLDER = DATA_TEST + "eaf/"
CSV_FOLDER = DATA_TEST + "csv/"

Converter(path_in=EAF_FOLDER, ext_in="eaf", path_out=SRT_FOLDER, ext_out="srt")