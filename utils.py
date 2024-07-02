import plotly.graph_objects as go
import numpy as np
from numpy import pi, sin, cos

def point_sphere(lon, lat):
    #associate the cartesian coords (x, y, z) to a point on the  globe of given lon and lat
    #lon longitude
    #lat latitude
    lon = lon*pi/180
    lat = lat*pi/180
    x = cos(lon) * cos(lat)
    y = sin(lon) * cos(lat)
    z = sin(lat)
    return np.array([x, y, z])

def shortest_path(A=[100, 45], B=[-50, -25], dir=-1, n=100):
    #Spherical "linear" interpolation
    """
    A=[lonA, latA] lon lat given in degrees; lon in  (-180, 180], lat in ([-90, 90])
    B=[lonB, latB]
    returns n points on the great circle of the globe that passes through the  points A, B
    #represented by lon and lat
    #if dir=1 it returns the shortest path; for dir=-1 the complement of the shortest path
    """
    As = point_sphere(A[0], A[1])
    Bs = point_sphere(B[0], B[1])
    alpha = np.arccos(np.dot(As,Bs)) if dir==1 else  2*pi-np.arccos(np.dot(As,Bs))
    if abs(alpha) < 1e-6 or abs(alpha-2*pi)<1e-6:
        return A
    else:
        t = np.linspace(0, 1, n)
        P = sin((1 - t)*alpha)
        Q = sin(t*alpha)
        #pts records the cartesian coordinates of the points on the chosen path
        pts =  np.array([a*As + b*Bs for (a, b) in zip(P,Q)])/sin(alpha)
        #convert cartesian coords to lons and lats to be recognized by go.Scattergeo
        lons = 180*np.arctan2(pts[:, 1], pts[:, 0])/pi
        lats = 180*np.arctan(pts[:, 2]/np.sqrt(pts[:, 0]**2+pts[:,1]**2))/pi
        return lons, lats

dict_names = {'Bahamas, The' : 'The Bahamas',
 'Congo, Dem. Rep.' : 'Democratic Republic of the Congo',
 "Cote d'Ivoire" : 'Ivory Coast',
 'Czech Republic' : 'Czechia',
 'Egypt, Arab Rep.' : 'Egypt',
 'Eswatini' : 'eSwatini',
 'Gambia, The' : 'Gambia',
 'Korea, Dem. Rep.' : 'North Korea',
 'Korea, Rep.' : 'South Korea',
 'Kyrgyz Republic' : 'Kyrgyzstan',
 'Russian Federation' : 'Russia',
 'Serbia' : 'Republic of Serbia',
 'Syrian Arab Republic' : 'Syria',
 'United States' : 'United States of America',
 'Venezuela, RB' : 'Venezuela',
 'West Bank and Gaza' : 'Palestine'}
