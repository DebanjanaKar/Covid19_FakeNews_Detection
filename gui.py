#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:50:48 2020

@author: suranjana
"""

import wx


########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.frame = parent
        
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        controlSizer = wx.BoxSizer(wx.VERTICAL)
        self.widgetSizer = wx.BoxSizer(wx.VERTICAL)
        
        self.text_ctrl = wx.TextCtrl(self)
        controlSizer.Add(self.text_ctrl, 0, wx.ALL | wx.EXPAND, 5)        
        my_btn = wx.Button(self, label='Detect Fake/NoFake')
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        controlSizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)  
        
        self.output = wx.StaticText(self, label='')
        controlSizer.Add(self.output, 1, wx.ALL, 5)
        
        
        self.mainSizer.Add(controlSizer, 0, wx.CENTER)
        self.mainSizer.Add(self.widgetSizer, 0, wx.CENTER|wx.ALL, 10)
        
        self.SetSizer(self.mainSizer)
        
        
    def on_press(self, event):
        value = self.text_ctrl.GetValue()
        self.onAddResult(value)#, self.artist)
        
    #----------------------------------------------------------------------
    def onAddResult(self, value):
        """"""
        classifier_output = value
        self.output.SetLabel(classifier_output)
        self.frame.fSizer.Layout()
        self.frame.Fit()
    
########################################################################
class MyFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, parent=None, title='Fake-OR-NoFake')
        self.fSizer = wx.BoxSizer(wx.VERTICAL)
        panel = MyPanel(self)
        self.fSizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(self.fSizer)
        self.Fit()
        self.Show()
        
#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()


