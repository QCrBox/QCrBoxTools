from qcrboxtools.robots.eval import PicFile

def test_string_representation_and_file_writing(tmp_path):
    content = "model 4\nxa 0.2\nxb 0.2\nxc 0.2\nlambdainit\nmicavec 0 0 1"
    pic_file = PicFile(content)
    file_path = tmp_path / "output.pic"
    pic_file.to_file(file_path)
    with open(file_path, 'r', encoding='UTF-8') as f:
        written_content = f.read()
    assert written_content.strip() == content.strip()


def test_command_parameters_parsing():
    content = "model 4\nxa 0.2 xb 0.2 xc 0.2\nlambdainit\nmicavec 0 0 1"
    pic_file = PicFile(content)
    assert pic_file['model'] == 4
    assert pic_file['xa'] == 0.2
    assert pic_file['xb'] == 0.2
    assert pic_file['xc'] == 0.2
    assert pic_file['lambdainit'] is None
    assert pic_file['micavec'][0] == 0
    assert pic_file['micavec'][1] == 0
    assert pic_file['micavec'][2] == 1
