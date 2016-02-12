# -*- coding: utf-8 -*-
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros

class STAT_PARSER:
    def __init__(self, tree_list):
        initial_root = tree_list[0].getroot()
        axis_xml = initial_root.find('STATISTICAL_DATA').find('CLASS_INF')
        self.axis_xml = axis_xml
        axis_list = []
        axis_len_list = []
        for axis in axis_xml:
            axis_id = axis.get('id')
            axis_name = axis.get('name')
            axis_obj_list = []
            axis_reverse_dict = {}
            for axis_obj in axis:
                axis_obj_list.append({'code': axis_obj.get('code'), 'name': axis_obj.get('name')})
                axis_reverse_dict[axis_obj.get('code')] = len(axis_obj_list) - 1
                print(str(len(axis_obj_list) - 1) + " " + axis_obj.get('code') + " " + axis_obj.get('name'))
            axis_list.append({'id': axis_id,
                              'name': axis_name,
                              'len': len(axis_obj_list),
                              'list': axis_obj_list,
                              'reverse_dict': axis_reverse_dict})
            axis_len_list.append(len(axis_obj_list))
        self.axis_list = axis_list
        table = np.zeros(axis_len_list)
        table.fill(np.nan)
        print(table.shape)
        for tree in tree_list:
            root = tree.getroot()
            data_xml = root.find('STATISTICAL_DATA').find('DATA_INF').findall('VALUE')
            for datum in data_xml:
                idx_list = []
                for axis in axis_list:
                    axis_id = axis['id']
                    code_in_this_axis = datum.get(axis['id'])
                    # print('code_in_this_axis', code_in_this_axis)
                    idx_list.append(axis['reverse_dict'][code_in_this_axis])
                    # print('idx_in_table', idx_in_table)
                # print(idx_list)
                try:
                    table[tuple(idx_list)] = int(datum.text)
                except ValueError:
                    table[tuple(idx_list)] = np.nan
            self.table = table
