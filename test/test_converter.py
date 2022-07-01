import os
import unittest
from main import *

DATA_TEST: str = "/home/jsanhcez/Documentos/Cursos/TensorBoard/scientificProject/data/data_test/"
JSON_FOLDER = DATA_TEST + "json/"
CTM_FOLDER = DATA_TEST + "ctm/"
STL_FOLDER = DATA_TEST + "stl/"
SRT_FOLDER = DATA_TEST + "srt/"
STM_FOLDER = DATA_TEST + "stm/"
TXT_FOLDER = DATA_TEST + "txt/"


class ConverterTestCase(unittest.TestCase):
    def test_json_to_ctm(self):
        ctm = Converter(JSON_FOLDER, "json", CTM_FOLDER, "ctm")


        self.assertIsNotNone(ctm)
        self.assertEquals(len(ctm), len(os.listdir(JSON_FOLDER)))

    def test_stl_to_srt(self):
        srt = Converter(STL_FOLDER, "stl", SRT_FOLDER, "srt")


        self.assertIsNotNone(srt)
        self.assertEquals(len(srt), len(os.listdir(SRT_FOLDER)))

    def test_srt_to_stm(self):
        stm = Converter(SRT_FOLDER, "srt", STM_FOLDER, "stm")


        self.assertIsNotNone(stm)
        self.assertEquals(len(stm), len(os.listdir(STM_FOLDER)))

    def test_srt_to_plain_transcription(self):
        txt = Converter(SRT_FOLDER, "srt", TXT_FOLDER, "txt")

        self.assertIsNotNone(txt)
        self.assertEquals(len(txt), len(os.listdir(STM_FOLDER)))


if __name__ == '__main__':
    unittest.main()
