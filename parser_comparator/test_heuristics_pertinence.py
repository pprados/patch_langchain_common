import unittest
import re

class TestListRegex(unittest.TestCase):

    def setUp(self):
        self.list_regex = re.compile(r"^([ \t]*)([-*+•◦▪·o]|\d+([./]|(\\.))) .+")
        self.test_dict = {
            "· Puce 5": "· Puce 5",
            "o Puce 2": "o Puce 2",
            "1/ Puce 1": "1/ Puce 1",
            "2/ Puce 2": "2/ Puce 2",
            "1\\. Puce 1": "1\\. Puce 1",
            "2\\. Puce 2": "2\\. Puce 2",
            "1. Puce 1": "1. Puce 1",
            "2. Puce 2": "2. Puce 2",
        }

    def test_list_regex(self):
        for name, string in self.test_dict.items():
            with self.subTest(name=name):
                match = self.list_regex.match(string)
                self.assertIsNotNone(match, f"Failed for {name}")

class TestTableRegex(unittest.TestCase):

    def setUp(self):
        self.regex_tables_html = r"(?s)(<table[^>]*>(?:.*?)<\/table>)"
        self.test_dict = {
            "html_table_1": """<table>
<tr>
<th>Le titre 1</th>
<th></th>
<th></th>
<th>Le titre 2</th>
</tr>
<tr>
<td colspan="2">Contenue de cell1</td>
<td></td>
<td>Contenue de cell2</td>
</tr>
<tr>
<td colspan="4">Une autre cellule 200.00</td>
</tr>
<tr>
<th>Sub cell</th>
<th>Sub cell2</th>
<th>Sub cell 3</th>
<th rowspan="2">Hello world, j'ai un texte sur plusieurs lignes et c'est cool</th>
</tr>
<tr>
<td>Dsq</td>
<td>Sqd</td>
<td>Dsq</td>
</tr>
</table>"""
        }

    def test_tables_html_regex(self):
        for name, html_table in self.test_dict.items():
            with self.subTest(name=name):
                match = re.match(self.regex_tables_html, html_table)
                self.assertIsNotNone(match, f"No matches found for {name}")

if __name__ == '__main__':
    unittest.main()