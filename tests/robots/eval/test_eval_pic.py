# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from qcrboxtools.robots.eval import PicFile

def test_string_representation_and_file_writing(tmp_path):
    content = "model 4\nxa 0.2\nxb 0.2\nxc 0.2\nlambdainit\nmicavec 0 0 1"
    pic_file = PicFile("output.pic", content)
    pic_file['micavec'][0].options[2] = 2
    pic_file.to_file(tmp_path)
    with open(tmp_path / "output.pic", 'r', encoding='UTF-8') as f:
        written_content = f.read()
    assert written_content.strip()[-1] == '2'
    assert written_content.strip()[:-1] == content.strip()[:-1]


def test_command_parameters_parsing():
    content = "model 4\nxa 0.2 xb 0.2 xc 0.2\nlambdainit\nmicavec 0 0 1"
    pic_file = PicFile('test.pic', content)
    assert pic_file['model'][0].options == 4
    assert pic_file['xa'][0].options == 0.2
    assert pic_file['xb'][0].options == 0.2
    assert pic_file['xc'][0].options == 0.2
    assert pic_file['lambdainit'][0].options is None
    assert pic_file['micavec'][0].options[0] == 0
    assert pic_file['micavec'][0].options[1] == 0
    assert pic_file['micavec'][0].options[2] == 1
