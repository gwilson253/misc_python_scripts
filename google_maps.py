# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:28:24 2019

@author: greg.wilson
"""

import os
import googlemaps

api_key = os.getenv('GOOGLE_MAPS_API_KEY')

gmaps = googlemaps.Client(key=api_key)

current_loc = '4700 NE 73rd Ave, 97218'
current_loc_geo = gmaps.geocode(current_loc)
current_loc_lat = current_loc_geo[0]['geometry']['location']['lat']
current_loc_lng = current_loc_geo[0]['geometry']['location']['lng']

search_for = 'Bank of America'
m_radius = 100000

loc_bias_template = 'circle:{rad}@{lat},{lng}'
loc_bias = loc_bias_template.format(rad=m_radius,
                                    lat=current_loc_lat,
                                    lng=current_loc_lng)

place_search_results = gmaps.find_place(search_for,
                                        input_type='textquery',
                                        location_bias=loc_bias)

place_1 = place_search_results['candidates'][0]
place_1 = gmaps.place(place_1['place_id'])

# get distance
dest_loc_lat = place_1['result']['geometry']['location']['lat']
dest_loc_lng = place_1['result']['geometry']['location']['lng']

dist_matrix = gmaps.distance_matrix((current_loc_lat, current_loc_lng),
                                    (dest_loc_lat, dest_loc_lng),
                                    mode='driving')