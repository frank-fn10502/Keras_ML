import configparser
from pathlib import Path


class Config:
    '''
    取得 trainig 的設定檔
    '''
    def __init__(self, path='cfg.ini') -> None:
        self.config = configparser.ConfigParser()
        self.config.read(path)

    def getCfgData(self, section, key, data = None):
        '''
        從 cfg 檔取得資料，和傳入的資料比對，如果不同將新資料紀錄於 self.config 以便之後 save 新資料
        - section: cfg 檔的區段
        - key: 區段中的哪一選項
        - data: 新資料，如果和 cfg 設定值不同則會自動更新值 

        exception KeyError:
            如果 section, key 非空則寫入新值
        '''
        result = None
        try:
            result = eval(self.config[section][key])

            if data is not None and data != result:
                self.config[section][key] = str(data)
                result = data

        except  KeyError:
            if section == None or key == None:
                raise Exception('請填入 section, key')

            self.config[section][key] = str(data)
            result = data

        return result

    def saveConfig(self, savePath: Path):
        '''
        訓練時將本次設定檔一併存在 rsult 資料夾，方便日後查看。
        '''
        with open(savePath / 'cfg.ini', 'w') as f:
            self.config.write(f)
    