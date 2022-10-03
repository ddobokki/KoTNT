import os
from os import PathLike


class TRNDataLoader:
    def __init__(self, raw_folder_path: PathLike) -> None:
        self.raw_folder_path = raw_folder_path
        self.scripts_folders = [file for file in os.listdir(raw_folder_path) if file.find("scripts") > -1]

    def generate(self):

        for scripts_folder in self.scripts_folders:
            raw_path = os.path.join(self.raw_folder_path, scripts_folder)
            trn_files = os.listdir(raw_path)
            for trn_file in trn_files:
                trn_file_path = os.path.join(raw_path, trn_file)
                file = open(trn_file_path)
                while True:
                    raw_line = file.readline()
                    if not raw_line:
                        break
                    try:
                        # TODO [청소, 바트] : trn 전처리간의 음성 처리는 빠질 것이므로, ::는 필요없게 바뀝니다. split할 필요는 없습니다. (앞으로)
                        sent = raw_line.split("::")[1]
                    except:
                        pass
                    yield sent
