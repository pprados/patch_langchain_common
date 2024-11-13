import unittest
from unittest.mock import Mock
from src.pdf_multi_parser import PDFMultiParser

class TestPDFMultiParser(unittest.TestCase):

    def test_partial_failure(self):
        parser1 = Mock()
        parser2 = Mock()
        parser3 = Mock()


        parser1.parse.side_effect = Exception("Parser 1 failed")
        doc_parser2 = Mock()
        doc_parser2.page_content = ""
        parser2.parse.return_value = [doc_parser2]
        parser3.parse.side_effect = Exception("Parser 3 failed")

        multi_parser = PDFMultiParser(parsers_dict=[parser1, parser2, parser3])

        blob = Mock()
        with self.assertLogs(level='WARNING') as lo:
            result = list(multi_parser.lazy_parse(blob))
            print(lo.output)

        self.assertEqual(len(result), 1)
        self.assertTrue(any("Parser 1 failed" in message for message in lo.output))
        self.assertTrue(any("Parser 3 failed" in message for message in lo.output))


    def test_all_failure(self):
        parser1 = Mock()
        parser2 = Mock()
        parser3 = Mock()

        parser1.parse.side_effect = Exception("Parser 1 failed.")
        parser2.parse.side_effect = Exception("Parser 2 failed.")
        parser3.parse.side_effect = Exception("Parser 3 failed.")

        multi_parser = PDFMultiParser(parsers_dict=[parser1, parser2, parser3])

        blob = Mock()
        try:
            list(multi_parser.lazy_parse(blob))
        except Exception as e:
            self.assertEqual(str(e), "All parsers have failed.")
        else:
            self.fail("An exception was expected, but none was raised.")

if __name__ == '__main__':
    unittest.main()
