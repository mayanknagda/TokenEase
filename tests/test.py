import unittest
from tokenease import Pipe


class TestPipe(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipe()

    def test_fit_transform(self):
        docs = ["This is a test document.", "This is another test document."]
        bow, docs = self.pipe.fit_transform(docs)
        self.assertEqual(bow.shape, (2, 9))  # 2 documents, 9 unique tokens

    def test_transform(self):
        docs = ["This is a test document.", "This is another test document."]
        self.pipe.fit_transform(docs)
        new_data = ["This is a new document."]
        bow, docs = self.pipe.transform(new_data)
        self.assertEqual(bow.shape, (1, 9))  # 1 document, 9 unique tokens

    def test_save_and_load(self):
        data = ["This is a test document.", "This is another test document."]
        self.pipe.fit_transform(data)
        self.pipe.save("test_pipe.joblib")
        loaded_pipe = Pipe.from_pretrained("test_pipe.joblib")
        self.assertEqual(loaded_pipe.vocabulary, self.pipe.vocabulary)


if __name__ == "__main__":
    unittest.main()
