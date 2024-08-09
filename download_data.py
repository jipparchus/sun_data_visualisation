import os
import requests

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.net import attrs as sun_attrs
from sunpy.net import Fido

"""
Please set the path to save the aia data.
!! Keep it consistent with HOME variable in main.py !!
Do NOT modify SAVE_TO.
"""
HOME = os.getcwd()
SAVE_TO = os.path.join(HOME, 'aia_data')


def download_dataset_from_jsoc(direc, email):
    os.makedirs(direc, exist_ok=True)
    for w in [171, 193, 304]:
        path_data = os.path.join(direc, f'aia{w}')
        os.makedirs(path_data, exist_ok=True)
        # Obtain AIA/SDO data from JSOC
        if len(os.listdir(path_data)) < 1:
            start_time = Time('2012-07-19T07:30:00', scale='utc', format='isot')
            bottom_left = SkyCoord(13.3 * u.arcmin, -6.7 * u.arcmin, obstime=start_time, observer="earth", frame="helioprojective")
            top_right = SkyCoord(17 * u.arcmin, -1.7 * u.arcmin, obstime=start_time, observer="earth", frame="helioprojective")
            cutout = sun_attrs.jsoc.Cutout(bottom_left=bottom_left, top_right=top_right, tracking=True)
            jsoc_email = email
            query = Fido.search(
                sun_attrs.Time(start_time, start_time + 15 * u.minute),
                sun_attrs.Wavelength(w * u.angstrom),
                sun_attrs.Sample(30 * u.second),
                sun_attrs.jsoc.Series.aia_lev1_euv_12s,
                sun_attrs.jsoc.Notify(jsoc_email),
                sun_attrs.jsoc.Segment.image,
                cutout,
            )
            print(query)
            Fido.fetch(query, path=path_data)
        else:
            pass


def download_dataset_from_github_repo(local_path=SAVE_TO, repo_path='aia_data'):
    """ Download a direc from Github repo. """
    api_url = f"https://api.github.com/repos/jipparchus/solar_visualisation/contents/{repo_path}"
    os.makedirs(local_path, exist_ok=True)
    response = requests.get(api_url)
    if response.status_code == 200:  # If success
        contents = response.json()
        for item in contents:
            print(item['type'], item['name'])
            if item['type'] == 'file':
                download_file(item['download_url'], os.path.join(local_path, item['name']))
            elif item['type'] == 'dir':  # If directory, go into the directory and download files inside.
                os.makedirs(os.path.join(local_path, item['name']), exist_ok=True)
                download_dataset_from_github_repo(os.path.join(local_path, item['name']), item['path'])
    else:
        print(f"Failed to access {api_url}: {response.status_code}")


def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:  # If success
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


if __name__ == '__main__':
    mode = 'github'
    if mode == 'github':
        download_dataset_from_github_repo(SAVE_TO)
    elif mode == 'jsoc':
        # Please register your email through the link: http://jsoc.stanford.edu/ajax/register_email_art.html
        download_dataset_from_jsoc(SAVE_TO, 'email@address')
