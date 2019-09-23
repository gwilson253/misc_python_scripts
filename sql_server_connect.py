# -*- coding: utf-8 -*-
"""
SQL Server Connect
April 2019 - Greg Wilson, Slalom
"""

import pyodbc
from pandas import read_sql
from sqlalchemy import create_engine
import urllib

class SqlServerConnect:
    '''
    This is a class for interfacing with SQL Server.

    Please do not change this code. This is meant to serve as a quick way to
    experiment with data in your SQL Server.

    Copy any of the code you would like to use into a new file.

    A production implementation of this class would require the following:
        - Implementing this code in a python package / module
        - Adding some sort of version control
        - Adding tests (e.g. pytest)
    '''

    def __init__(self, server='', database=''):
        '''
        Returns a SqlServerConnect object.

        Args:
            :server (str): server name; Default is ''
            :database (str): database name; Default is ''
        '''
        self.server = server
        self.database = database

    def get_connection(self):
        '''Returns a pyodbc connection object'''
        return pyodbc.connect(driver='{SQL Server}',
                              server=self.server,
                              database=self.database,
                              trusted_connection=True)

    def get_sql_engine(self):
        '''Returns a SQLAlchemy engine'''
        template = 'DRIVER={{SQL Server Native Client 10.0}};SERVER={};DATABASE={};trusted_connection=yes'
        params = urllib.parse.quote_plus(template.format(self.server, self.database))
        return create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    def read_sql(self, sql):
        '''Takes a SQL SELECT statement and returns a pandas DataFrame'''
        with self.get_connection() as conn:
            return read_sql(sql, conn)

    def write_df(self, df, tablename, schema=None, if_exists='fail'):
        '''
        Replicates the pandas.DataFrame.to_sql function.

        :df (pandas.DataFrame): DataFrame to be written
        :tablename (str): The name of the target table
        :schema (str): The schema of the target table
        :if_exists: How to behave if the table already exists.
            * fail (default): raise a ValueError
            * replace: drop the table before inserting new values
            * append: insert new values to the existing table
        '''
        df.to_sql(name=tablename,
                  con=self.get_sql_engine(),
                  schema=schema,
                  if_exists=if_exists,
                  index=False)

    def execute_sql(self, sql):
        '''
        Executes a SQL command.
        '''
        with self.get_connection() as conn:
            try:
                conn.execute(sql)
            except Exception as e:
                print('SQL Execution failed!\n', e)
