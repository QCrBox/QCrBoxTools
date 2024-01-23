import pytest
from pathlib import Path
from qcrboxtools.robots.eval import TextFile

@pytest.fixture
def sample_file(tmp_path):
    sample_text = "This is a sample text."
    file = tmp_path / "sample.txt"
    file.write_text(sample_text, encoding='UTF-8')
    return file

def test_reading_file(sample_file):
    text_file = TextFile.from_file(str(sample_file))
    assert text_file.text == "This is a sample text."
    assert text_file.filename == "sample.txt"

def test_writing_file(tmp_path, sample_file):
    text_file = TextFile.from_file(str(sample_file))
    new_content = "This is a new text."
    text_file.text = new_content
    text_file.to_file(str(tmp_path))
    new_file_path = tmp_path / "sample.txt"
    assert new_file_path.read_text(encoding='UTF-8') == new_content

def test_writing_to_different_directory(tmp_path, sample_file):
    text_file = TextFile.from_file(str(sample_file))
    new_content = "Modified text."
    text_file.text = new_content
    new_directory = tmp_path / "subfolder"
    new_directory.mkdir()
    text_file.to_file(str(new_directory))
    new_file_path = new_directory / "sample.txt"
    assert new_file_path.read_text(encoding='UTF-8') == new_content

def test_writing_to_current_directory(sample_file, monkeypatch):
    text_file = TextFile.from_file(str(sample_file))
    new_content = "Content for current directory."
    text_file.text = new_content
    with monkeypatch.context() as m:
        m.chdir(sample_file.parent)
        text_file.to_file()
        assert Path("sample.txt").read_text(encoding='UTF-8') == new_content
