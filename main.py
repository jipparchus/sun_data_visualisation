import base64
from datetime import datetime
from io import BytesIO
import numpy as np
import os
import tempfile

import astropy.units as u
from astropy.io import fits
import cv2
from flask import Flask
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import requests
from scipy.interpolate import splprep, splev
from sunpy.coordinates import frames as sunframes
import sunpy.map
from sunpy.visualization.colormaps import color_tables as ct
from tqdm import tqdm

HOME = os.getcwd()  # Please set the project homepath
WAVELENGTHS = [171, 193, 304]
AIACMAP = {w: ct.aia_color_table(w * u.AA) for w in WAVELENGTHS}
POINTS = np.array([(258.8, 320.3), (242.0, 305.2), (233.6, 296.2), (223.5, 273.9), (218.5, 258.2), (215.1, 226.8), (216.3, 209.5), (215.7, 189.3)])
mpl.rc('figure.subplot', left=0.12, right=0.95, bottom=0.2, top=0.9)


def download_dataset_from_github_repo(local_path=os.path.join(HOME, 'aia_data'), repo_path='aia_data'):
    """ Download a direc from Github repo. """
    api_url = f"https://api.github.com/repos/jipparchus/solar_visualisation/contents/{repo_path}"
    os.makedirs(local_path, exist_ok=True)
    response = requests.get(api_url)
    if response.status_code == 200:  # If success
        contents = response.json()
        for item in tqdm(contents, desc="Downloading observation data ...", unit_scale=True):
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
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def pxl2heliocoord(sunpymap, pxl_coord=(0, 0)):
    """ Transformation from pixel coordinates Helioprojective coordinates at each point """
    phys_coord = sunpymap.pixel_to_world(pxl_coord[1] * u.pix, pxl_coord[0] * u.pix)
    helioprojective_coord = phys_coord.transform_to(sunframes.Helioprojective(observer=sunpymap.observer_coordinate))
    lon, lat = helioprojective_coord.Tx, helioprojective_coord.Ty
    return (lon.value, lat.value)


def interpolate_points():
    """ Get B-spline coords for interpolated points. """
    tck, u = splprep([POINTS[:, 0], POINTS[:, 1]], s=0)
    u_new = np.linspace(u.min(), u.max(), 1000)
    return splev(u_new, tck)


def cut_along_curve(img, x, y, width=5):
    """ Cutout image along a given curve and returns square cutout image. """
    height, width_img = img.shape
    cutout = []
    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / length, dy / length  # Unit vector along path
        for j in range(-width, width):
            new_x, new_y = x1 - uy * j, y1 + ux * j
            new_x, new_y = np.clip(new_x, 0, width_img - 1), np.clip(new_y, 0, height - 1)
            if len(cutout) <= j + width:
                cutout.append([])
            cutout[j + width].append(img[int(new_y), int(new_x)])
    return np.array(cutout).T


def to_html_image(fig):
    buf = BytesIO()
    fig.savefig(buf, dpi=200, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


class DataSet():
    def __init__(self, wavelength=304):
        self.wavelength = int(wavelength)
        self.cmap = AIACMAP[self.wavelength]
        self.folder = os.path.join(HOME, 'aia_data', f'aia{wavelength}')
        self.load_fits()
        self.dates = [hdr['DATE-OBS'] for hdr in self.headers]
        self.dates_datetime = [datetime.strptime(dd, '%Y-%m-%dT%H:%M:%S.%f') for dd in self.dates]
        self.get_blurred()  # Remove large bright structures from image at each time fram and enhance the loop structures. Plot in log10 scale
        self.data_enhanced = np.log10(self.data / self.data_blurred)
        self.get_bi()
        self.get_time_distance()

    def load_fits(self):
        """ Load FITS files """
        files = [os.path.join(self.folder, f) for f in os.listdir(self.folder)]
        # number of time frames
        self.nt = len(files)
        self.headers = []
        self.sunpymaps = []
        self.extents = []
        for i, file in enumerate(files):
            with fits.open(file) as hdul:
                hdul.verify('fix')
                data, header = hdul[1].data, hdul[1].header
                sunpymap = sunpy.map.Map(file)
                self.sunpymaps.append(sunpymap)
                if i == 0:
                    # shape of obs data. Data is stored in a shape [height, width]
                    self.ny, self.nx = data.shape
                    # Save data as (x,y,t) shape
                    self.data = np.zeros((self.nx, self.ny, self.nt))

                ny, nx = sunpymap.data.shape
                corners_pixel_dict = {'bl': (0, 0), 'tl': (ny - 1, 0), 'br': (0, nx - 1), 'tr': (ny - 1, nx - 1)}
                corners_helioprojective_dict = {}
                for key, coord in corners_pixel_dict.items():
                    corners_helioprojective_dict[key] = pxl2heliocoord(sunpymap, coord)
                # Helioprojective coordinates of corner points of the field-of-view
                left = np.mean([corners_helioprojective_dict['bl'][0], corners_helioprojective_dict['tl'][0]])
                right = np.mean([corners_helioprojective_dict['br'][0], corners_helioprojective_dict['tr'][0]])
                top = np.mean([corners_helioprojective_dict['tl'][1], corners_helioprojective_dict['tr'][1]])
                bottom = np.mean([corners_helioprojective_dict['bl'][1], corners_helioprojective_dict['br'][1]])
                self.extents.append((left, right, bottom, top))

                # Normalise the image pixel values by the exposure time
                if header['PIXLUNIT'] == 'DN':
                    data = data / header['EXPTIME']
                    header['PIXLUNIT'] = 'DN/S'
                # Intensity must be positive float. Replacce zero or negative values with minimum non-zero pixel value
                data_pos = np.where(data <= 0, np.nan, data).reshape(-1, 1)
                pos_values = data_pos[~np.isnan(data_pos)]
                data = np.where(data <= 0, pos_values.min(), data)
                # Transpose the data as the original data has shape (height,width) while self.data has shape (width,height,time)
                self.data[:, :, i] = data.T
                self.headers.append(header)

    def get_blurred(self):
        """ Gaussian blur filter """
        self.data_blurred = np.zeros_like(self.data)
        for i in range(self.nt):
            self.data_blurred[:, :, i] = cv2.GaussianBlur(self.data[:, :, i], ksize=(51, 51), sigmaX=-1)

    def get_bi(self, ksize=(3, 3)):
        """ Create binary skeleton mask to emphasise the structures in images. """
        kernel = np.ones(ksize, np.uint8)
        self.data_bi = np.zeros_like(self.data)
        for i in range(self.nt):
            # Gaussian blur
            blur = cv2.GaussianBlur(self.data_enhanced[:, :, i], (7, 7), 0)
            # Convert float pixvalue to uint8
            blur = ((blur - blur.min()) / (blur.max() - blur.min()) * 255).astype('uint8')
            # img_bi = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
            img_bi = np.where(blur > np.percentile(blur, 70), 255, 0).astype('uint8')
            # Morphological opening and closing
            img_opn = cv2.morphologyEx(img_bi, cv2.MORPH_OPEN, kernel)
            self.data_bi[:, :, i] = cv2.morphologyEx(img_opn, cv2.MORPH_CLOSE, kernel)

    def generate_gif(self, *args):
        """ Make animation of original, blurred, and enhanced images """
        nimg = len(args)
        global_min = []
        global_max = []
        for j in range(nimg):
            global_min.append(np.percentile(args[j], 0.2))
            global_max.append(np.percentile(args[j], 99.8))
        # Set up the animation environment
        fig, ax = plt.subplots(1, nimg, figsize=(7, 3), sharex='all', sharey='all', facecolor='whitesmoke')
        fig.text(0.5, 0.0125, 'Solar X (arcsec)', ha='center', va='bottom', fontsize=10)
        fig.text(0.0125, 0.5, 'Solar Y (arcsec)', ha='left', va='center', rotation='vertical', fontsize=10)
        # kwargs for imshow

        def imshow_kwgs(j):
            return {'cmap': self.cmap, 'origin': 'lower', 'norm': Normalize(vmin=global_min[j], vmax=global_max[j])}
        imgs = []
        # Initialise the frame
        for j in range(nimg):
            img = ax[j].imshow(args[j][:, :, 0].T, extent=self.extents[0], **imshow_kwgs(j))
            imgs.append(img)

        def update(i):
            _imgs = []
            for j in range(nimg):
                if j == 0:
                    ax[j].set_title(f"{self.dates[i].split('.')[0]} (UT)", fontsize='large')
                imgs[j].set_data(args[j][:, :, i].T)
                imgs[j].set_extent(self.extents[i])
                imgs[j].set_clim(vmin=global_min[j], vmax=global_max[j])
                _imgs.append(imgs[j])
            return tuple(_imgs)
        # Create the animation object
        ani = animation.FuncAnimation(fig, update, frames=self.nt, interval=100, repeat_delay=None, blit=True)
        # Save the animation to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_file:
            writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
            ani.save(temp_file.name, writer=writer, dpi=200)
            # Read the file contents into a BytesIO object
            temp_file.seek(0)
            buf = BytesIO(temp_file.read())
        # buf contains the gif data, buf.getvalue() to access it.
        os.remove(temp_file.name)
        return buf

    def get_time_distance(self):
        xx, yy = interpolate_points()
        for i in range(self.nt):
            if i == 0:
                nalong, _ = cut_along_curve(self.data_enhanced[:, :, i], xx, yy).shape
                self.time_distance = np.zeros((self.nt, nalong))
            self.time_distance[i, :] = np.mean(cut_along_curve(self.data_enhanced[:, :, i], xx, yy), axis=1)
        self.extent_td = (self.dates_datetime[0], self.dates_datetime[-1], 0, nalong - 1)
        return self.time_distance, self.extent_td


def get_len_km(ds: DataSet, dist_arcsec):
    """Convert arcsec distance to distance in km by small angle approximation"""
    sun_dist = ds.headers[0]['DSUN_OBS']  # distance to the Sun in meters
    return np.pi / 180 / 3600 * sun_dist * dist_arcsec / 1e3


if __name__ == '__main__':
    if not os.path.exists(os.path.join(HOME, 'aia_data')):
        download_dataset_from_github_repo()
    # Show the plots on browser
    app = Flask(__name__)

    @app.route("/")
    def index():
        # Create dataset instances
        list_ds = [ds171, ds193, ds304] = [DataSet(w) for w in WAVELENGTHS]
        # Helioprojective coordinate of example cutout path which is interpolated from POINTS.
        xx_helio, yy_helio = pxl2heliocoord(ds171.sunpymaps[0], interpolate_points())
        p0, p1 = np.array([xx_helio[0], yy_helio[0]]), np.array([xx_helio[-1], yy_helio[-1]])
        # Get the approximated length of the cutout path as L2 norm of p1-p0 vector
        dist_km = get_len_km(ds171, np.linalg.norm(p1 - p0, ord=2))

        c = '#1de95b'
        fig, axes = plt.subplots(1, 2, sharex='all', sharey='all')
        for ax, img, cmap in zip(axes.ravel(), [ds171.data_bi[:, :, 0], ds171.data_enhanced[:, :, 0]], ['gray', ds171.cmap]):
            ax.set_xlabel('Solar X (arcsec)', fontsize=10)
            ax.set_ylabel('Solar Y (arcsec)', rotation='vertical', fontsize=10)
            ax.imshow(img.T, origin='lower', extent=ds171.extents[0], cmap=cmap, alpha=0.7)
            ax.plot(xx_helio, yy_helio, ls='--', alpha=0.9, c=c)
            ax.scatter(xx_helio[0], yy_helio[0], marker='o', s=20, ec=c, fc='none')
            ax.scatter(xx_helio[-1], yy_helio[-1], marker='s', s=20, ec=c, fc='none')
            ax.annotate('P0', p0, xytext=(0, -18), textcoords='offset points', bbox=dict(boxstyle="round", ec=c, fc="none"), c=c)
            ax.annotate('P1', p1, xytext=(0, -18), textcoords='offset points', bbox=dict(boxstyle="round", ec=c, fc="none"), c=c)
        html_path = fr'<h2>Sample path selection at 171 {u.AA} {ds171.dates[0]} (UT)</h2><img src={to_html_image(fig)}>'

        fig, axes = plt.subplots(3, 1, sharex='all', sharey='all')
        fig.text(0.5, 0.0125, 'Date (UT)', ha='center', va='bottom', fontsize=10)
        fig.text(0.0125, 0.5, 'P0 to P1 Distance (km)', ha='left', va='center', rotation='vertical', fontsize=10)
        t0, t1 = ds304.dates_datetime[10], ds304.dates_datetime[20]
        x0, x1 = 5 * 1e3, 25 * 1e3
        for ax, ds in zip(axes.ravel(), list_ds):
            _ax = ax.twinx()
            _ax.set_yticks([0, round(dist_km)])
            _ax.set_yticklabels(['P0', 'P1'])
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.imshow(ds.time_distance, cmap=ds.cmap, origin='lower', extent=[ds.extent_td[0], ds.extent_td[1], 0, round(dist_km)], aspect='auto')
            ax.plot((t0, t1), (x0, x1), c='#4169e1', ls=':', lw=2, alpha=0.8)
            ax.annotate(f'{round((x1 - x0) / (t1 - t0).seconds)} km/s', (t1, x1), xytext=(0, 0), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="w"), c='#4169e1')
            ax.set_xlim(ds304.dates_datetime[0], ds304.dates_datetime[-1])
        html_cutout = fr'<h2>Cutout time-distance plots at (171, 193, 304 {u.AA})</h2><img src={to_html_image(fig)}>'

        head1 = 'Solar Dynamics Observatory (SDO)'
        head2 = 'Atmospheric Imaging Assembly (AIA) data visualisation'
        html_0 = '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">'
        html_head = f'{html_0}<title>earthwave Python Challenge</title></head><body><h1>{head1}</h1><h1>{head2}</h1><article>'
        html_imgs = ''
        for w, ds in zip(WAVELENGTHS, list_ds):
            buf = ds.generate_gif(ds.data, ds.data_blurred, ds.data_enhanced)
            src = f"data:image/gif;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            html_imgs += fr"<h2>{str(w)} {u.AA}</h2><h3>Original, Blurred, Loop Enhanced</h3><img src={src}>"
        html_comment = f"<h3>* Flows of plasma bulbs ('coronal rain') are observed to have speeds around {round((x1 - x0) / (t1 - t0).seconds)} km/s.</h3>"
        html_tail = '</article></body></html>'
        return html_head + html_imgs + html_path + html_cutout + html_comment + html_tail
    app.run(debug=False)
