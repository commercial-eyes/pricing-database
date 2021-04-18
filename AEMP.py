import pandas as pd
import numpy as np
import json
from tqdm.notebook import tqdm
import time
from zipfile import ZipFile
from io import BytesIO
from sys import exit

import requests
from requests_html import HTML

from pathlib import Path
from urlpath import URL
import string
  

def timeis(func):
    '''Decorator that reports the execution time.'''
  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__name__, end-start)
        return result
    
    return wrap


def read_download_log(BASE_DIR):
    """
    Returns the logged HREFs that have already been imported into the
    pricing database.
    """
    
    path = Path(BASE_DIR) / 'logs' / 'download_log.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, 'r') as log:
            return json.load(log)
    else:
        return []

    
def write_download_log(BASE_DIR, log_data):
    """
    Write the HREFs that have been imported into the pricing database
    to the download log.
    """
    
    path = Path(BASE_DIR) / 'logs' / 'download_log.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as log:
        json.dump(log_data, log)
    
    
def check_for_updates(BASE_DIR, exman_prices_url):
    """
    Checks the current exman-prices download page for new data that isn't stored in the
    download log. Returns a list of hrefs that need to be imported.
    """
    
    download_log = read_download_log(BASE_DIR)
    r = requests.get(exman_prices_url)
    html = HTML(html=r.content)

    dataset_urls = [ele.attrs['href'] for ele in html.find('a') 
                    if (ele.attrs['href'].lower().__contains__('xls')) and
                    (ele.attrs['href'] not in download_log)]
    
    return dataset_urls


def read_exman_source(url):
    """
    Reads source exman .xls file, including date and schedule from url,
    and returns a cleaned df 
    """

    def read_exman_date(url):
        """
        Returns the date of the dataset passed in the url in datetime format
        """

        url = URL(url)

        return pd.to_datetime('-'.join(url.stem.split('-')[-3:]))


    def read_exman_schedule(url):
        """
        Returns the schedule of the dataset passed in the url (efc or non-efc)
        """

        url = URL(url)

        if url.stem.partition('prices-')[2].partition('-')[0] == 'efc':
            return 'efc'
        else:
            return 'non-efc'

    date = read_exman_date(url)
    schedule = read_exman_schedule(url)
    df = pd.read_excel(url)
    df['Date'] = date
    df['Schedule'] = schedule

    return clean_df(df, schedule)

class AMTNameError(Exception):
    pass


def clean_df(df, schedule):
    """
    Cleans df, with different actions based on schedule  
    """

    
    def catch_amt_name(col):
        """
        Used to catch the AMT trade product pack (TPP) name for standardisation,
        as it changes across time
        """
        
        catch = ('amt', 'tpp', 'product pack')
        return any(x in col.lower() for x in catch)
    
    
    df.columns = [col.strip() for col in df.columns] # clean whitespace
    
    AMT_name = list(filter(catch_amt_name, df.columns))
    if not AMT_name:
        raise AMTNameError(f'AMT name not found')
    if len(AMT_name) > 1:
        raise AMTNameError('len(AMT_name) > 1')
    
    AMT_name = AMT_name[0]
    df.rename(columns={AMT_name: 'AMT TPP'}, inplace=True)
        
    if 'C\'wlth Pays Premium' in df.columns:
        df.rename(columns={'C\'wlth Pays Premium': 'Commonwealth Pays Premium'}, inplace=True)

    rename_cols = {
        'Maximum Amount': 'Maximum Quantity/Amount',
        'Maximum Quantity': 'Maximum Quantity/Amount',
        'Number Repeats': 'Maximum/Number Repeats',
        'Maximum Repeats': 'Maximum/Number Repeats',
        'DPMA': 'DPMQ/DPMA',
        'DPMQ':'DPMQ/DPMA',
        'Claimed DPMA': 'Claimed DPMQ/DPMA',
        'Claimed DPMQ':'Claimed DPMQ/DPMA'
    }
    
    rename_cols = {k: v for k, v in rename_cols.items() if k in df.columns}
    df.rename(columns=rename_cols, inplace=True)

    # depreceated/misnamed columns to drop
    columns_to_drop = [
        'index', 'AMT Trade Product Pack Pack', 'Exempt', 'Therapeutic Group',
        'New PI or Brand', 'Previous Pricing Quantity', 'Previous AEMP', 'Price Change Event',
        'Previous Premium', 'ATC', 'DD', 'MRVSN', 'Substitutable', ' Item Code',
        'Authorised Rep', 'Email', 'AMT Trade Product pack', 'AMT Trade product Pack',
        'ANT Trade Product Pack', 'TPP', 'AMT Trade Product Pack ', 'Amt Trade Product Pack'
    ]
        
    columns_to_drop = [field for field in filter(columns_to_drop.__contains__, df.columns)]
    df.drop(columns_to_drop, axis=1, inplace=True)
        
    # set commonwealth pays premium to a boolean
    if 'Commonwealth Pays Premium' in df.columns:
        df['Commonwealth Pays Premium'] = df['Commonwealth Pays Premium'].apply(
            lambda x: True if x == 'Yes' else False
        )
    
    # clean up premium type and premium value
    float_chars = '1234567890.'
    if 'Premium' in df.columns:
        df['Premium Type'] = df['Premium'].apply(lambda x: ''.join(filter(string.ascii_letters.__contains__, str(x))))
        df['Premium'] = df['Premium'].apply(lambda x: ''.join(filter(float_chars.__contains__,str(x))))

    return df


def download_updates(BASE_DIR, exman_prices_url):
    """
    Downloads any missing datasets from the PBS and returns these as a df
    """

    dataset_urls = check_for_updates(BASE_DIR, exman_prices_url)
    if not dataset_urls:
        return None

    new_data = []
    for url in tqdm(dataset_urls, desc='Importing new data'):
        new_data.append(read_exman_source(BASE_URL / url))

    df = pd.concat(new_data, sort=False)
    
    # log updates
    log = read_download_log(BASE_DIR)
    log.extend(dataset_urls)
    write_download_log(BASE_DIR, log)
    
    return df.reset_index(drop=True)
    

def load_db(BASE_DIR, latest_month_only=False, name='db'):
    """
    Loads local pricing database
    """
    
    path = BASE_DIR / name
    if not path.exists():
        return None
    
    df = pd.read_feather(path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    if latest_month_only:
        df = df.loc[df.Date == df.Date.max()]
    
    return df

def write_db(BASE_DIR, df, append=False, name='db'):
    """
    Writes dataframe to local pricing database, or appends to existing
    """
    
    df.reset_index(drop=True, inplace=True)
    
    if not append:
        df.to_feather(BASE_DIR / 'db')
    else:
        df_base = load_db(BASE_DIR)
        df = pd.concat([df_base, df], sort=False)
        df.reset_index(drop=True).to_feather(BASE_DIR / 'db')

@timeis
def perform_lookup(df, item_map, atc_map):
    """
    Generate Unique SKU IDs for generating previous AEMPs and creating longitudinal relationships
    """

    lookup_cols = [
        'Item Code', 'AMT TPP', 'Pack Quantity', 'Pricing Quantity', 
        'Vial Content', 'Maximum Quantity/Amount', 'Number/Maximum Repeats'
    ]

    # filter to cols that are present in df
    print('Creating lookup column...')
    lookup_cols = [col for col in lookup_cols if col in df.columns]
    df['SKU ID'] = df[lookup_cols].applymap(str).agg('-'.join,axis=1)

    # generate previous AEMPs
    print('Calculating previous prices...')
    df['Previous AEMP'] = df.groupby('SKU ID')['AEMP'].shift(fill_value=np.nan)

    # indicators of increase/decrease for analysis
    conditions_increase = [
      (df['AEMP'] > df['Previous AEMP']) & (df['Previous AEMP'] != np.nan),
      (df['AEMP'] <= df['Previous AEMP']) & (df['Previous AEMP'] != np.nan)
    ]
    conditions_decrease = [
      (df['AEMP'] < df['Previous AEMP']) & (df['Previous AEMP'] != np.nan),
      (df['AEMP'] >= df['Previous AEMP']) & (df['Previous AEMP'] != np.nan)
    ]
    outcomes = [True, False]

    # numpy - return 1,0 for increase/decrease conditionals
    df['AEMP_Increase'] = np.select(conditions_increase, outcomes, default=False)
    df['AEMP_Decrease'] = np.select(conditions_decrease, outcomes, default=False)

    # absolute and relative change calculations
    df['AEMP_Abs_Change'] = df['AEMP'] - df['Previous AEMP']
    df['AEMP_Rel_Change'] = (df['AEMP'] - df['Previous AEMP']) / df['Previous AEMP']

    # generate 6 dig item code for map lookup
    print('Generating ATC labels...')
    
    df['ItemCodeLookup'] = df['Item Code'].map(lambda x: '0'*(6-len(str(x)))+str(x))

    df = df.merge(item_map[['ITEM_CODE','ATC_Code']], how='left', left_on='ItemCodeLookup', right_on='ITEM_CODE').drop('ITEM_CODE',axis=1)
    df.rename(columns={'ATC5_Code':'ATC_Code'},inplace=True)

    # create levelled ATC codes
    df['ATC1'] = df['ATC_Code'].str[0]
    df['ATC3'] = df['ATC_Code'].str[:3]
    df['ATC4'] = df['ATC_Code'].str[:4]
    df['ATC5'] = df['ATC_Code'].str[:5]

    for ATC_level in tqdm(['ATC1','ATC3','ATC4','ATC5','ATC_Code']):
        df = df.merge(atc_map[['ATC Code', 'Label']],
                      how='left',
                      left_on=ATC_level,
                      right_on='ATC Code').drop('ATC Code', axis=1)
        df.rename(columns={'Label': ATC_level + '_label'}, inplace=True)

    df.drop('ItemCodeLookup',axis=1,inplace=True)

    return df

class PBSData:
    """
    Wrapper for methods to grab various data from the PBS website
    """
    
    def __init__(self):
        self.BASE_DIR = Path(r'C:\Python\AEMP')
        self.BASE_URL = URL('https://www.pbs.gov.au')
        self.source_url = BASE_URL / 'info/browse/download'
        self.item_drug_map_url = BASE_URL / '/statistics/dos-and-dop/files/pbs-item-drug-map.csv'
        self.exman_prices_url = BASE_URL / 'info/industry/pricing/ex-manufacturer-price'

        
    def get_latest_PBS_text_files(self):
        """
        Returns a ZipFile of the most recent PBS text files
        """

        r = requests.get(self.source_url)
        html = HTML(html=r.content)

        # filter to current PBS text files .zip
        href = [ele for ele in html.find('a.xref') 
                         if ('PBS Text files' in ele.attrs['title'])
                         and ('.zip' in ele.attrs['href'].lower())][0].attrs['href']
        discard, sep, url = href.partition('downloads')
        zip_url = sep + url

        r = requests.get(BASE_URL / zip_url, stream=True)
        return ZipFile(BytesIO(r.content))


    def get_atc_from_text_files(self, zipfile):
        """
        Gets ATC code map from PBS text files and returns as df with cols ('ATC', 'Description')
        """

        zipfile_dir = zipfile.namelist()
        for file in zipfile_dir:
            if 'atc_' in file:
                break

        with zipfile.open(file) as f:
            data = [line.decode('utf-8').strip().split('!') for line in f.readlines()][1:]

        return pd.DataFrame(data, columns=['ATC Code', 'Label'])


    def get_item_drug_map(self):
        """
        Returns a df of the PBS item drug map
        """


        def strip_leading_zeros(x):
            """
            Strips leading zeros from item code
            """

            while x[0] == '0':
                x = x[1:]
            return x

        df = pd.read_csv(self.item_drug_map_url, encoding='latin-1')
        df.columns = ['ITEM_CODE', 'DRUG_NAME', 'PRESENTATION', 'ATC_Code']

        return df
    