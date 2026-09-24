import gc
import os
from pydoc import locate
# import plotting as plot
import numpy as np
import ROOT
import copy as cp 
import sys
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

# plot.ModTDRStyle()

def convert_graph_to_th1(name,graph,template_hist):
    #make a copy of the template histogram
    hist = cp.deepcopy(template_hist)
    for i in range(graph.GetN()):
        #hist.Fill(graph.GetX()[i], graph.GetY()[i])
        #fill the histogram using set bin content
        hist.SetBinContent(i+1, graph.GetY()[i])
        #set error
        hist.SetBinError(i+1, graph.GetErrorY(i))  

    return hist


def __overflow_last_bin(hist, region, xmax=1200):
    nbins = hist.GetNbinsX()
    
    if region in ["C", "D"]:
        overflow_content = 0
        overflow_error_sq = 0
        
        for i in range(2, nbins + 1):
            overflow_content += hist.GetBinContent(i)
            overflow_error_sq += hist.GetBinError(i) ** 2
        
        hist.SetBinContent(1, hist.GetBinContent(1) + overflow_content)
        hist.SetBinError(1, np.sqrt(hist.GetBinError(1) ** 2 + overflow_error_sq))
        
        rebin_factor = nbins
        hist_rebinned = hist.Rebin(rebin_factor, hist.GetName() + "_rebinned")
        
        # print(f"Region {region}: Collapsed to first bin with total content {hist_rebinned.GetBinContent(1):.2f}")
        return hist_rebinned
    
    last_bin = -1
    for i in range(1, nbins + 1):
        bin_center = hist.GetBinCenter(i)
        if bin_center <= xmax:
            last_bin = i
        else:
            break
    
    if last_bin == -1 or last_bin == nbins:
        # print(f"Region {region}: No overflow needed (last_bin={last_bin}, nbins={nbins})")
        return hist
    
    overflow_content = 0
    overflow_error_sq = 0
    
    for i in range(last_bin + 1, nbins + 1):
        overflow_content += hist.GetBinContent(i)
        overflow_error_sq += hist.GetBinError(i) ** 2
    
    # print(f"Region {region}: Overflowing {overflow_content:.2f} events into bin {last_bin} at xmax={xmax}")
    
    hist.SetBinContent(last_bin, hist.GetBinContent(last_bin) + overflow_content)
    hist.SetBinError(last_bin, np.sqrt(hist.GetBinError(last_bin) ** 2 + overflow_error_sq))
    
    bins_to_merge = nbins - last_bin + 1
    if bins_to_merge > 1:
        bin_edges = [hist.GetBinLowEdge(i) for i in range(1, last_bin + 1)]
        bin_edges.append(hist.GetBinLowEdge(last_bin + 1))
        hist_rebinned = hist.Rebin(len(bin_edges) - 1, hist.GetName() + "_rebinned", np.array(bin_edges, dtype=float))
    else:
        hist_rebinned = hist.Clone(hist.GetName() + "_rebinned")
    hist_rebinned.SetDirectory(0)
    hist_rebinned.Sumw2()
    return hist_rebinned

def __param_config_to_human_readable(param_config):
    text = param_config
    splits = text.split("_")
    text = text.replace("_", ", ")
    for split in splits:
        new_split = split
        if "mMed" in new_split:
            new_split = new_split.replace("mMed", "m_{#Phi} = ")
            new_split = new_split + " GeV"
        elif "mZp" in new_split:
            new_split = new_split.replace("mZp", "m_{Z'} = ")
            new_split = new_split + " GeV"
        elif "rinv" in new_split:
            new_split = new_split.replace("rinv", "r_{inv} = ")
            new_split = new_split.replace("p", ".")
        elif "mDark" in new_split:
            new_split = new_split.replace("mDark", "m_{dark} = ")
            new_split = new_split + " GeV"
        elif "yukawa" in new_split:
            new_split = new_split.replace("yukawa", "#lambda = ")
            new_split = new_split.replace("p", ".")
        text = text.replace(split, new_split)
    
    if text.endswith(" "): text = text[:-1]
    return text

def __fix_hist_from_roofit(histogram, histname = ""):
    reweighting_factors = []
    # histogram.Sumw2()
    print(f" --------------- Fixing histogram {histogram.GetName()} for RooFit bin merging effects.")
    min_bin_width = min([histogram.GetBinWidth(i) for i in range(histogram.GetNbinsX())])
    for i in range(1, histogram.GetNbinsX()+1):
        bin_width = histogram.GetBinWidth(i)
        bin_value = histogram.GetBinContent(i)
        bin_error = histogram.GetBinError(i)
        
        histogram.SetBinContent(i, bin_value * bin_width / min_bin_width)
        histogram.SetBinError(i, bin_error * bin_width / min_bin_width)
        reweighting_factors.append(bin_width / min_bin_width)
        # if "C" in histname or "D" in histname:
        #     # For regions C and D, we expect the histogram to be collapsed to one bin, so we skip the reweighting
        #     print(f"{histname} values are. - bin {i}: content = {bin_value:.8f}, error = {bin_error:.8f}, width = {bin_width:.8f}, reweighting factor = {bin_width / min_bin_width:.8f}, bin content after reweighting = {histogram.GetBinContent(i):.8f}, bin error after reweighting = {histogram.GetBinError(i):.8f}")
    
    return reweighting_factors


def __get_params_dict(fit_results, what, categories_channels_mapping,category):
    """Return parameter dictionary.

    Args:
        fit_results (str): ROOT file with fit results
        what (str): "prefit", "fit_b" or "fit_s"
    """

    fin = ROOT.TFile(fit_results)
    norm = fin.Get("norm_" + what)

    channel_year_region = categories_channels_mapping[category]
    #get all channels from the mapping and initialize the dictionary params_dict
    params_dict = {}
    channels = []
    for element in channel_year_region:
        params_dict[element[0]] = {}
        channels.append(element[0])


    it = norm.createIterator()
    while True:
        try:
            var = it.Next()
            if not var:
                break
            # print("variable name: ",  var.GetName())
            if var.GetName().split("/")[1] == "total_background" and var.GetName().split("/")[0] in channels:
                params_dict[var.GetName().split("/")[0]]["Bkg"] = var.getVal()
            elif var.GetName().split("/")[0] in channels and var.GetName().split("/")[1] == "total_signal":
                # print("Normalization: ", var.GetName())
                params_dict[var.GetName().split("/")[0]]["Sig"] = var.getVal()
        except:
            #restart the iterator
            print("Restarting the iterator !!!")
            it = norm.createIterator()
    
    return params_dict


def __get_signal_histograms(fit_results, what,  categories_channels_mapping,category,abcd_opt):
    fin = ROOT.TFile(fit_results)

    first_dir = "shapes_" + what

    #get from categories_channels_mapping[category] the corresponding channel
    channel_year_region = categories_channels_mapping[category]
    #loop over elements in channel_year_region, and assign second_dir_A to the channel for which the region is A, etc
    for element in channel_year_region:
        if element[2] == "A":
            second_dir_A = element[0]
        elif element[2] == "B":
            second_dir_B = element[0]
        elif element[2] == "C":
            second_dir_C = element[0]
        elif element[2] == "D":
            second_dir_D = element[0]

    h_sig_A  = fin.Get(first_dir + "/" + second_dir_A + "/total_signal")
    h_sig_B  = fin.Get(first_dir + "/" + second_dir_B + "/total_signal")
    h_sig_C  = fin.Get(first_dir + "/" + second_dir_C + "/total_signal")
    h_sig_D  = fin.Get(first_dir + "/" + second_dir_D + "/total_signal")

    __fix_hist_from_roofit(h_sig_A)
    __fix_hist_from_roofit(h_sig_B)
    __fix_hist_from_roofit(h_sig_C)
    __fix_hist_from_roofit(h_sig_D)


    params_dict = __get_params_dict(fit_results, what, categories_channels_mapping,category)

    #print integral 
    print("total signal normalization in SR: " , params_dict[second_dir_A]["Sig"])

    h_sig_A.Scale(params_dict[second_dir_A]["Sig"]/h_sig_A.Integral())
    h_sig_B.Scale(params_dict[second_dir_B]["Sig"]/h_sig_B.Integral())
    h_sig_C.Scale(params_dict[second_dir_C]["Sig"]/h_sig_C.Integral())
    h_sig_D.Scale(params_dict[second_dir_D]["Sig"]/h_sig_D.Integral())

    h_sig_A = __overflow_last_bin(h_sig_A, "A")
    h_sig_B = __overflow_last_bin(h_sig_B, "B")
    h_sig_C = __overflow_last_bin(h_sig_C, "C")
    h_sig_D = __overflow_last_bin(h_sig_D, "D")

    histograms = {
        "A": h_sig_A.Clone(),
        "B": h_sig_B.Clone(),
        "C": h_sig_C.Clone(),
        "D": h_sig_D.Clone(),
    }
    #apply remapping of the regions
    if abcd_opt.remap_back_postfit:
        #use abcd_opt.map_regions to remap the regions
        histograms_remapped = {}
        for region in ["A", "B", "C", "D"]:
            histograms_remapped[region] = histograms[abcd_opt.map_regions[region]]

    else:
        histograms_remapped = histograms

    return histograms_remapped

def __get_bkg_histograms(fit_results, what, categories_channels_mapping,category,abcd_opt):
    """Return parameter dictionary.

    Args:
        fit_results (str): ROOT file with fit results
        postfit_params_dict
        what (str): "prefit", "fit_b" or "fit_s"
    """

    fin = ROOT.TFile(fit_results)

    first_dir = "shapes_" + what

    #get from categories_channels_mapping[category] the corresponding channel
    channel_year_region = categories_channels_mapping[category]
    #loop over elements in channel_year_region, and assign second_dir_A to the channel for which the region is A, etc
        
    for element in channel_year_region:
        if element[2] == "A":
            second_dir_A = element[0]
        elif element[2] == "B":
            second_dir_B = element[0]
        elif element[2] == "C":
            second_dir_C = element[0]
        elif element[2] == "D":
            second_dir_D = element[0]

    h_bkg_A  = fin.Get(first_dir + "/" + second_dir_A + "/total_background")
    h_bkg_B  = fin.Get(first_dir + "/" + second_dir_B + "/total_background")
    h_bkg_C  = fin.Get(first_dir + "/" + second_dir_C + "/total_background")
    h_bkg_D  = fin.Get(first_dir + "/" + second_dir_D + "/total_background")

    # Correct for the fact that bins "merged" together are averaged out instead of being summed up in RooFit
    # print(f"channel_year_region: {channel_year_region}, first_dir: {first_dir}, second_dir_A: {second_dir_A}")
    # print(f"h_bkg_A before fix: {h_bkg_A.Integral()}")
  
    h_bkg_A.SetDirectory(0)
    h_bkg_B.SetDirectory(0)
    h_bkg_C.SetDirectory(0)
    h_bkg_D.SetDirectory(0)
    h_bkg_A.Sumw2()
    h_bkg_B.Sumw2()
    h_bkg_C.Sumw2()
    h_bkg_D.Sumw2()

    __fix_hist_from_roofit(h_bkg_A)

    # print("************** Fixing background histograms for RooFit bin merging effects... **************")
    # print(f"category: {category}, region B before fix: integral = {h_bkg_B.Integral():.2f}, do what - {what}")
    
    # print(f"Sumw2N for B - {h_bkg_B.GetSumw2N()}")
    # for bin in range(1, h_bkg_B.GetNbinsX() + 1):
    #     print(f"Bin {bin}: content = {h_bkg_B.GetBinContent(bin):.2f}, error = {h_bkg_B.GetBinError(bin):.2f}, width = {h_bkg_B.GetBinWidth(bin):.2f}")        
    __fix_hist_from_roofit(h_bkg_B)
    # print(f"category: {category}, region B after fix: integral = {h_bkg_B.Integral():.2f}")
    # for bin in range(1, h_bkg_B.GetNbinsX() + 1):
    #     print(f"Bin {bin}: content = {h_bkg_B.GetBinContent(bin):.2f}, error = {h_bkg_B.GetBinError(bin):.2f}, width = {h_bkg_B.GetBinWidth(bin):.2f}")
    __fix_hist_from_roofit(h_bkg_C, histname = f"{category}_C")
    __fix_hist_from_roofit(h_bkg_D)

    #scale the shapes to the prefit normalization
    params_dict = __get_params_dict(fit_results, what, categories_channels_mapping,category)
    if h_bkg_A.Integral() > 0:
        h_bkg_A.Scale(params_dict[second_dir_A]["Bkg"]/h_bkg_A.Integral())
    else:
        h_bkg_A.Scale(params_dict[second_dir_A]["Bkg"])

    if h_bkg_B.Integral() > 0:
        h_bkg_B.Scale(params_dict[second_dir_B]["Bkg"]/h_bkg_B.Integral())
    else:
        h_bkg_B.Scale(params_dict[second_dir_B]["Bkg"])

    if h_bkg_C.Integral() > 0:
        print("************** Scaling background histogram C to postfit normalization... **************")
        print(f"category: {category}, region C before scaling: integral = {h_bkg_C.Integral():.8f}, do what - {what}, postfit normalization = {params_dict[second_dir_C]['Bkg']:.8f}, second_dir_C: {second_dir_C}, params_dict[second_dir_C]: {params_dict[second_dir_C]}")
        h_bkg_C.Scale(params_dict[second_dir_C]["Bkg"]/h_bkg_C.Integral())
        print("************** After scaling background histogram C **************")
        print(f"category: {category}, region C after scaling: integral = {h_bkg_C.Integral():.8f}, do what - {what}, scaling factor = {params_dict[second_dir_C]['Bkg']/h_bkg_C.Integral():.8f}")
    else:
        h_bkg_C.Scale(params_dict[second_dir_C]["Bkg"])

    if h_bkg_D.Integral() > 0:
        h_bkg_D.Scale(params_dict[second_dir_D]["Bkg"]/h_bkg_D.Integral())
    else:
        h_bkg_D.Scale(params_dict[second_dir_D]["Bkg"])

    h_bkg_A = __overflow_last_bin(h_bkg_A, "A")
    h_bkg_B = __overflow_last_bin(h_bkg_B, "B")
    h_bkg_C = __overflow_last_bin(h_bkg_C, "C")
    h_bkg_D = __overflow_last_bin(h_bkg_D, "D")

    histograms = {
        "A": h_bkg_A.Clone(),
        "B": h_bkg_B.Clone(),
        "C": h_bkg_C.Clone(),
        "D": h_bkg_D.Clone(),
    }
    #apply remapping of the regions
    if abcd_opt.remap_back_postfit:
        #use abcd_opt.map_regions to remap the regions
        histograms_remapped = {}
        for region in ["A", "B", "C", "D"]:
            histograms_remapped[region] = histograms[abcd_opt.map_regions[region]]

    else:
        histograms_remapped = histograms
    

    return histograms_remapped


def __get_data_histograms(fit_results, data_yields_dict, what, categories_channels_mapping,category,abcd_opt):

    fin = ROOT.TFile(fit_results)

    first_dir = "shapes_" + what

   #get from categories_channels_mapping[category] the corresponding channel
    channel_year_region = categories_channels_mapping[category]
    #loop over elements in channel_year_region, and assign second_dir_A to the channel for which the region is A, etc
    for element in channel_year_region:
        if element[2] == "A":
            second_dir_A = element[0]
        elif element[2] == "B":
            second_dir_B = element[0]
        elif element[2] == "C":
            second_dir_C = element[0]
        elif element[2] == "D":
            second_dir_D = element[0]

    h_dat_A  = fin.Get(first_dir + "/" + second_dir_A + "/data")  # This is a TGraphAsymmErrors, not a TH1F
    h_dat_B  = fin.Get(first_dir + "/" + second_dir_B + "/data")  # This is a TGraphAsymmErrors, not a TH1F
    h_dat_C  = fin.Get(first_dir + "/" + second_dir_C + "/data")  # This is a TGraphAsymmErrors, not a TH1F
    h_dat_D  = fin.Get(first_dir + "/" + second_dir_D + "/data")  # This is a TGraphAsymmErrors, not a TH1F

    h_bkg_A  = fin.Get(first_dir + "/" + second_dir_A + "/total_background")
    h_bkg_B  = fin.Get(first_dir + "/" + second_dir_B + "/total_background")
    h_bkg_C  = fin.Get(first_dir + "/" + second_dir_C + "/total_background")
    h_bkg_D  = fin.Get(first_dir + "/" + second_dir_D + "/total_background")

    #convert the data graph to a TH1F using a template histogram
    h_dat_A = convert_graph_to_th1(h_dat_A.GetName(),h_dat_A.Clone(),h_bkg_A.Clone())
    h_dat_B = convert_graph_to_th1(h_dat_B.GetName(),h_dat_B.Clone(),h_bkg_B.Clone())
    h_dat_C = convert_graph_to_th1(h_dat_C.GetName(),h_dat_C.Clone(),h_bkg_C.Clone())
    h_dat_D = convert_graph_to_th1(h_dat_D.GetName(),h_dat_D.Clone(),h_bkg_D.Clone())
    
    # Correct for the fact that bins "merged" together are averaged out instead of being summed up in RooFit
    __fix_hist_from_roofit(h_dat_A)
    __fix_hist_from_roofit(h_dat_B)
    __fix_hist_from_roofit(h_dat_C)
    __fix_hist_from_roofit(h_dat_D)

    #scale the data to the prefit normalization
    h_dat_A.Scale((data_yields_dict[second_dir_A])/(h_dat_A.Integral()))
    h_dat_B.Scale((data_yields_dict[second_dir_B])/(h_dat_B.Integral()))
    h_dat_C.Scale((data_yields_dict[second_dir_C])/(h_dat_C.Integral()))
    h_dat_D.Scale((data_yields_dict[second_dir_D])/(h_dat_D.Integral()))

    h_dat_A = __overflow_last_bin(h_dat_A, "A")
    h_dat_B = __overflow_last_bin(h_dat_B, "B")
    h_dat_C = __overflow_last_bin(h_dat_C, "C")
    h_dat_D = __overflow_last_bin(h_dat_D, "D")
    

    histograms = {
        "A": cp.deepcopy(h_dat_A.Clone()),
        "B": cp.deepcopy(h_dat_B.Clone()),
        "C": cp.deepcopy(h_dat_C.Clone()),
        "D": cp.deepcopy(h_dat_D.Clone()),
    }
    #apply remapping of the regions
    if abcd_opt.remap_back_postfit:
        #use abcd_opt.map_regions to remap the regions
        histograms_remapped = {}
        for region in ["A", "B", "C", "D"]:
            histograms_remapped[region] = histograms[abcd_opt.map_regions[region]]

    else:
        histograms_remapped = histograms

    return histograms_remapped

def __read_data_yields(ws_fitdiagnostics_results,inject_signal_in_asimov,param_config, categories_channels_mapping,category):
    ws_tfile = ROOT.TFile(ws_fitdiagnostics_results)
    ws = ws_tfile.Get("w")

    #set as keys of data_yields_dict the channels in the categories_channels_mapping
    list_of_channels_years_regions = categories_channels_mapping[category]
    channels = [x[0] for x in list_of_channels_years_regions]

    data_yields_dict = {}
    for channel in channels:
        data_yields_dict[channel] = 0


    params_snapshot = ws.getSnapshot("clean")
    it_snapshot = params_snapshot.createIterator()

    while True:
        var = it_snapshot.Next()
        if not var:
            break

        if "shapeBkg_data_obs" in var.GetName():
            data_yields_dict[var.GetName().split("_")[3]] = var.getVal() 

    if inject_signal_in_asimov:
        #get all formulas 
        embedded_all_data = ws.allEmbeddedData()
        #loop over RooArgSet elements in formulas
        
        for ch in channels:
            #loop over the formulas and get the value of the signal yield
            for embedded_data in embedded_all_data:
                    if "shapeSig_%s_%s"%(param_config,ch) in embedded_data.GetName():
                        # print("Signal yield for %s: %f"%(ch,embedded_data.sumEntries()))
                        data_yields_dict[ch] += embedded_data.sumEntries()
                        
    return data_yields_dict


 def __combine_histograms_years(histograms_cat_years, what, flavor):
     histograms_combined = {r: None for r in ["A", "B", "C", "D"]}
     for region in ["A", "B", "C", "D"]:
         region_hists = []
         for year in histograms_cat_years.keys():
             hist = histograms_cat_years[year][what + "_" + flavor][region]
             region_hists.append(hist)
         min_nbins = min(h.GetNbinsX() for h in region_hists)
         reference_hist = next(h for h in region_hists if h.GetNbinsX() == min_nbins)
         bin_edges = [reference_hist.GetBinLowEdge(i) for i in range(1, min_nbins + 1)]
         bin_edges.append(reference_hist.GetBinLowEdge(min_nbins + 1))
         rebinned_hists = []
         for hist in region_hists:
             if hist.GetNbinsX() != min_nbins:
                 hist_rebinned = hist.Rebin(len(bin_edges) - 1, hist.GetName() + "_rebinned", np.array(bin_edges, dtype=float))
                 rebinned_hists.append(hist_rebinned)
             else:
                 rebinned_hists.append(hist)
         for hist in rebinned_hists:
             if histograms_combined[region] is None:
                 histograms_combined[region] = cp.deepcopy(hist)
             else:
                 histograms_combined[region].Add(hist)
+
+        if flavor == "data":
+            for bin_idx in range(1, histograms_combined[region].GetNbinsX() + 1):
+                total = histograms_combined[region].GetBinContent(bin_idx)
+                histograms_combined[region].SetBinError(
+                    bin_idx,
+                    np.sqrt(max(0.0, total))
+                )
+
     return histograms_combined


# def rebin_hist_to_edges(hist, bin_edges):
#     from ROOT import TH1F
#     # Create new histogram with desired bin edges
#     name = hist.GetName() + "_rebinned"
#     new_hist = TH1F(name, hist.GetTitle(), len(bin_edges)-1, np.array(bin_edges, dtype=float))
#     # Fill new histogram with contents from old histogram
#     for i in range(1, hist.GetNbinsX()+1):
#         bin_center = hist.GetBinCenter(i)
#         bin_content = hist.GetBinContent(i)
#         bin_error = hist.GetBinError(i)
#         # Find which bin in new_hist this center falls into
#         bin_idx = new_hist.FindBin(bin_center)
#         # Add content and error (quadrature) to the new bin
#         new_hist.SetBinContent(bin_idx, new_hist.GetBinContent(bin_idx) + bin_content)
#         new_hist.SetBinError(bin_idx, np.sqrt(new_hist.GetBinError(bin_idx)**2 + bin_error**2))
#     return new_hist

def check_fit_results(root_file):
    f = ROOT.TFile(root_file)
    if f.IsZombie():
        print("Error: fit results file is corrupted: ", root_file)
        return []
    keys = f.GetListOfKeys()
    if keys.GetSize() == 0:
        print("Error: fit results file is empty: ", root_file)
        return []
    
    key_names = [key.GetName() for key in keys]
    available_fits = []
    
    for fit_type in ["fit_b", "fit_s", "prefit"]:
        if any(keyname == fit_type or keyname.startswith(f"shapes_{fit_type}") for keyname in key_names):
            available_fits.append(fit_type)
        else:
            print(f"Warning: fit results file does not contain {fit_type}: {root_file}")
    
    return available_fits

# plotting scripts for the stitched region plots using mplhep, adapted from the ABCD_plots.py script.
def __mpl_year_lumi(year):
    year = str(year)
    lumi_map = {"2016": 36.3, "2017": 42.1, "2018": 59.6, "Run2": 138}
    return lumi_map.get(year, None)


def __th1_to_arrays(h):
    n = h.GetNbinsX()
    v = np.array([h.GetBinContent(i) for i in range(1, n + 1)], dtype=float)
    e = np.array([h.GetBinError(i) for i in range(1, n + 1)], dtype=float)
    return v, e


def __build_uniform_edges(categories, cat_bins, region):
    edges = [0.0]
    for cat, ncb in zip(categories, cat_bins):
        for _ in range(ncb):
            width = 3.0 if (cat == "3PSVJ" and ncb == 1 and region in ["A", "B"]) else 1.0
            edges.append(edges[-1] + width)
    return np.array(edges, dtype=float)



def __build_xlabels_from_hists(hists, skip_x_label=2):
    labels = []
    for h in hists:
        # is_3psvj = ("3PSVJ" in h.GetName())
        for i in range(1, h.GetNbinsX() + 1):
            # if is_3psvj or i % 2 != 0:
            if (i) % skip_x_label != 0:
                labels.append(str(int(h.GetXaxis().GetBinLowEdge(i))))
            else:
                labels.append("")
    last_h = hists[-1]
    nbins = last_h.GetNbinsX()
    # is_3psvj = ("3PSVJ" in last_h.GetName())
    # if is_3psvj or (nbins + 1) % 2 != 0:
    if (nbins + 1) % 2 != 0:
        labels.append(str(int(last_h.GetXaxis().GetBinUpEdge(nbins))))
    else:
        labels.append("")
    
    return labels


def __stitch_hists_to_uniform_bins(hist_list, categories, cat_bins, region, out_name):
    import array as _array
    uniform_edges = __build_uniform_edges(categories, cat_bins, region)
    h_out = ROOT.TH1D(out_name, "", len(uniform_edges) - 1, _array.array("d", uniform_edges.tolist()))
    h_out.SetDirectory(0)
    h_out.Sumw2()

    out_bin = 1
    for h in hist_list:
        for i in range(1, h.GetNbinsX() + 1):
            h_out.SetBinContent(out_bin, h.GetBinContent(i))
            h_out.SetBinError(out_bin, h.GetBinError(i))
            out_bin += 1

    return h_out, uniform_edges


def __collect_stitched_region_payload(histograms_by_category, region, bkg_key, data_key, sig_key_map, category_order=None, skip_x_label=2):
    categories = category_order if category_order is not None else sorted(histograms_by_category.keys())

    kept_categories, cat_bins = [], []
    bkg_hists, data_hists = [], []
    sig_hists = {lbl: [] for lbl in sig_key_map.keys()}

    for cat in categories:
        if cat not in histograms_by_category:
            continue
        hdict = histograms_by_category[cat]
        if bkg_key not in hdict or data_key not in hdict:
            continue
        if region not in hdict[bkg_key] or region not in hdict[data_key]:
            continue

        h_bkg = hdict[bkg_key][region].Clone(f"{bkg_key}_{cat}_{region}")
        h_dat = hdict[data_key][region].Clone(f"{data_key}_{cat}_{region}")
        h_bkg.SetDirectory(0)
        h_dat.SetDirectory(0)

        kept_categories.append(cat)
        cat_bins.append(h_bkg.GetNbinsX())
        bkg_hists.append(h_bkg)
        data_hists.append(h_dat)

        for lbl, skey in sig_key_map.items():
            if skey in hdict and region in hdict[skey] and hdict[skey][region]:
                hs = hdict[skey][region].Clone(f"{skey}_{cat}_{region}")
                hs.SetDirectory(0)
            else:
                hs = h_bkg.Clone(f"zero_{lbl}_{cat}_{region}")
                hs.Reset("ICESM")
                hs.SetDirectory(0)
            sig_hists[lbl].append(hs)

    if not kept_categories:
        return None

    return {
        "categories": kept_categories,
        "cat_bins": cat_bins,
        "bkg_hists": bkg_hists,
        "data_hists": data_hists,
        "sig_hists": sig_hists,
        "x_labels": __build_xlabels_from_hists(data_hists,skip_x_label=skip_x_label),
    }

def __draw_stitched_plot(payload, outname, year, fit_label, param_config, ratio_label="Data/Bkg.", draw_sb_line=False, add_yields_to_legend=False, tagger="Supervised", cms_label=""):
    from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator


    categories = payload["categories"]
    cat_bins = payload["cat_bins"]
    x_labels = payload["x_labels"]

    region_id = fit_label.split(",")[0].split()[-1]
    h_bkg, uniform_edges = __stitch_hists_to_uniform_bins(
        payload["bkg_hists"], categories, cat_bins, region_id, f"h_bkg_{os.path.basename(outname)}"
    )
    h_data, _ = __stitch_hists_to_uniform_bins(
        payload["data_hists"], categories, cat_bins, region_id, f"h_data_{os.path.basename(outname)}"
    )

    stitched_sigs = {}
    for lbl, hlist in payload["sig_hists"].items():
        h_sig, _ = __stitch_hists_to_uniform_bins(
            hlist, categories, cat_bins, region_id, f"h_sig_{lbl}_{os.path.basename(outname)}"
        )
        stitched_sigs[lbl] = h_sig

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    bkg_label = f"Background ({h_bkg.Integral():.2e})" if add_yields_to_legend else "Background"
    if "prefit" in fit_label:
        bkg_label = f"Background (Data-driven $N_A = N_B × N_C / N_D$) ({h_bkg.Integral():.2e})" if add_yields_to_legend else "Background (Data-driven $N_A = N_B × N_C / N_D$)"
    hep.histplot([h_bkg], stack=True, histtype="fill", edgecolor="black", color="#64C0E8",
                 label=[bkg_label], ax=ax)

    bkg_v, bkg_e = __th1_to_arrays(h_bkg)
    x_centers = uniform_edges[:-1] + np.diff(uniform_edges) / 2.0
    ax.bar(x_centers, 2 * bkg_e, bottom=bkg_v - bkg_e, width=np.diff(uniform_edges),
           alpha=0.5, color="grey", hatch="////", label="Background uncertainty")

    data_label = f"Data ({h_data.Integral():.2e})" if add_yields_to_legend else "Data"
    hep.histplot(h_data, yerr=True, histtype="errorbar", color="black",
                 markersize=8, label=data_label, ax=ax)

    hmax = max(
        h_bkg.GetMaximum() + h_bkg.GetBinError(h_bkg.GetMaximumBin()),
        h_data.GetMaximum() + h_data.GetBinError(h_data.GetMaximumBin()),
    )

    if "Signal" in stitched_sigs and stitched_sigs["Signal"].Integral() > 0:
        signal_text = __param_config_to_human_readable(param_config).replace("#", "\\").replace("2000 GeV","2 TeV").replace("GeV", "\\text{ GeV}").replace("TeV", "\\text{ TeV}")
        sig_label = f"Signal ({stitched_sigs['Signal'].Integral():.2e})" if add_yields_to_legend else f"Signal (${signal_text}$)"
        hep.histplot(stitched_sigs["Signal"], histtype="step", yerr=False, color="red", linewidth=2,
                     label=sig_label, ax=ax)
        hmax = max(hmax, stitched_sigs["Signal"].GetMaximum())

    if "Signal × μ" in stitched_sigs and stitched_sigs["Signal × μ"].Integral() > 0:
        mu_label = f"Signal × μ ({stitched_sigs['Signal × μ'].Integral():.2e})" if add_yields_to_legend else "Signal × μ"
        hep.histplot(stitched_sigs["Signal × μ"], histtype="step", yerr=False, color="#f89c20", linestyle="--", linewidth=2,
                     label=mu_label, ax=ax)
        hmax = max(hmax, stitched_sigs["Signal × μ"].GetMaximum())

    if draw_sb_line and "Signal × μ" in stitched_sigs:
        h_sb = h_bkg.Clone(f"h_sb_{os.path.basename(outname)}")
        h_sb.Add(stitched_sigs["Signal × μ"])
        sb_label = f"S × μ + B ({h_sb.Integral():.2e})" if add_yields_to_legend else "S × μ + B"
        hep.histplot(h_sb, histtype="step", yerr=False, color="green", linewidth=2,
                     label=sb_label, ax=ax)
        hmax = max(hmax, h_sb.GetMaximum())

    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.set_ylim(0.1, max(1.0, hmax) * 1e4)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2.0, 10.0), numticks=100))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(NullLocator())
    # ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    # ax.minorticks_off()

    # for label, tick in zip(x_labels, ax.xaxis.get_major_ticks()):
    #     if label != '':
    #         length = 16  # longer tick
    #     else:
    #         length = 8  # shorter tick
    #     tick.tick1line.set_markersize(length)
    #     tick.tick2line.set_markersize(length)


    # cat_lbl_map = {c: c.replace("SVJ", " $n_{SVJ}^{pn}$").replace("P", "+") for c in categories}
    if tagger == "Unsupervised":
        cat_lbl_map = {
            "0SVJ": "$n_{SVJ}^{WNAE} = 0$",
            "1SVJ": "$n_{SVJ}^{WNAE} = 1$",
            "2SVJ": "$n_{SVJ}^{WNAE} = 2$",
            "3PSVJ": "$n_{SVJ}^{WNAE} \geq 3$"
        }
    else:
        cat_lbl_map = {
            "0SVJ": "$n_{SVJ}^{PN} = 0$",
            "1SVJ": "$n_{SVJ}^{PN} = 1$",
            "2SVJ": "$n_{SVJ}^{PN} = 2$",
            "3PSVJ":"$n_{SVJ}^{PN} \geq 3$"
        }
    cum_bin_idx = 0
    for i, (cat, ncb) in enumerate(zip(categories, cat_bins)):
        if i > 0:
            x = uniform_edges[cum_bin_idx]
            ymax = max(1.0, hmax) * 8
            ax.vlines(x, ymin=0.1, ymax=ymax, color="black", linestyle="--", alpha=0.7)
            rax.axvline(x, color="black", linestyle="--", alpha=0.7)
        local = uniform_edges[cum_bin_idx:cum_bin_idx + ncb + 1]
        ax.text((local[0] + local[-1]) / 2.0, max(1.0, hmax) * 5, cat_lbl_map.get(cat, cat),
                ha="center", va="top", fontsize=14, fontweight="bold")
        cum_bin_idx += ncb

    h_ratio = h_data.Clone(f"h_ratio_{os.path.basename(outname)}")
    h_ratio.Divide(h_bkg)
    hep.histplot(h_ratio, yerr=True, histtype="errorbar", color="black", markersize=8, ax=rax)

    bkg_rel_err = np.divide(bkg_e, bkg_v, out=np.zeros_like(bkg_e), where=bkg_v != 0)
    ones = np.ones_like(bkg_rel_err)
    print(f"DEBUG: fit label - {fit_label}, year - {year}, X centers: {x_centers}, bkg values: {bkg_v}, bkg errors: {bkg_e}, bkg relative errors: {bkg_rel_err}")

    rax.bar(x_centers, 2 * bkg_rel_err, bottom=ones - bkg_rel_err, width=np.diff(uniform_edges),
            alpha=0.5, color="grey", hatch="////")

    if draw_sb_line and "Signal × μ" in stitched_sigs:
        h_sb_ratio = h_bkg.Clone(f"h_sb_ratio_{os.path.basename(outname)}")
        h_sb_ratio.Add(stitched_sigs["Signal × μ"])
        h_sb_ratio.Divide(h_bkg)
        hep.histplot(h_sb_ratio, histtype="step", color="green", linewidth=2, ax=rax)

    rax.axhline(1.0, color="black", ls="--", lw=1)
    rax.set_ylim(0.01, 2.0)
    rax.set_ylabel(ratio_label, y=0.5, labelpad=35)
    rax.set_xlabel("$p_T^{miss}$ [GeV]")
    # rax.grid(True, which="major", linestyle="--", linewidth=0.5)
    rax.yaxis.set_minor_locator(AutoMinorLocator(5))
    rax.minorticks_on()
    rax.xaxis.set_minor_locator(NullLocator())
    # rax.minorticks_off()
    
    rax.set_xlim(uniform_edges[0], uniform_edges[-1])

    rax.set_xticks(uniform_edges)
    # rax.set_xticklabels(x_labels, rotation=75, fontsize=14, )
    if region_id in ["A", "B"]:
        x_labels = [
        '' if (i > 0 and x_labels[i] != '' and x_labels[i-1] != '') else x_labels[i]
        for i in range(len(x_labels))
        ]
    x_labels[-1] = '5000'
    rax.set_xticklabels(x_labels, fontsize=17, )
    
    for tick_ax in [ax, rax]:
        for label, tick in zip(x_labels, tick_ax.xaxis.get_major_ticks()):
            length = 16 if label != '' else 8
            tick.tick1line.set_markersize(length)
            tick.tick2line.set_markersize(length)

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", alpha=1)
    
    # signal_text = __param_config_to_human_readable(param_config).replace("#", "\\").replace("2000 GeV","2 TeV").replace("GeV", "\\text{ GeV}").replace("TeV", "\\text{ TeV}")
    # signal_text_parts = signal_text.split(", ")
    # if len(signal_text_parts) > 2:
    #     signal_text = ", ".join(signal_text_parts[:2]) + " $,\n$ " + ", ".join(signal_text_parts[2:])
    print("DEBUG: fit label - ", fit_label)
    # combined_text = f"{fit_label}\n$ {signal_text} $"
    combined_text = fit_label.replace(", ","\n") + f"\n{tagger}"
    ax.text(0.02, 0.96, combined_text, transform=ax.transAxes, va="top", ha="left", fontsize=21, bbox=bbox_props)

    handles, labels = ax.get_legend_handles_labels()
    if any("Data" in lbl for lbl in labels):
        idx = next(i for i, lbl in enumerate(labels) if "Data" in lbl)
        h0, l0 = handles.pop(idx), labels.pop(idx)
        handles.insert(0, h0)
        labels.insert(0, l0)
    # if "b-only" in fit_label:
    #     legend_fontsize, legend_title_fontsize = 28, 18
    # else:
    #     legend_fontsize, legend_title_fontsize = 15, 17
    ax.legend(handles, labels, ncol=max(2, len(labels) // 3), fontsize=17, loc="upper right", frameon=True, edgecolor="white", framealpha=1)
    # print(f"DEBUG: Lumi value - {__mpl_year_lumi(year)} for year {year}")
    hep.cms.label(f"{cms_label}", data=True, ax=ax, lumi=__mpl_year_lumi(year))
    fig.savefig(f"{outname}.pdf", bbox_inches="tight")
    print(f"Saved plot: {outname}.pdf")
    plt.close(fig)

def save_prefit_stitched_plots(histograms_by_category, dir_name, param_config, year, category_order=None, tagger="Supervised", cms_label=""):
    output_dir = os.path.join(dir_name, "prefit_stitched")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if tagger == "Supervised":
        skip_x_label = 2
    elif tagger == "Unsupervised":
        skip_x_label = 4

    for region in ["A", "B", "C", "D"]:
        payload = __collect_stitched_region_payload(
            histograms_by_category=histograms_by_category,
            region=region,
            bkg_key="prefit_bkg",
            data_key="prefit_data",
            sig_key_map={"Signal": "prefit_signal"},
            category_order=category_order,
            skip_x_label=skip_x_label,
        )
        if payload is None:
            continue

        outname = os.path.join(output_dir, f"plot_prefit_stitched_{region}_{year}")
        __draw_stitched_plot(
            payload=payload,
            outname=outname,
            year=year,
            fit_label=f"Region {region}, prefit",
            param_config=param_config,
            ratio_label="Data/Bkg.",
            draw_sb_line=False,
            tagger=tagger,
            cms_label=cms_label,
        )


def save_postfit_b_stitched_plots(histograms_by_category, dir_name, param_config, year, category_order=None, tagger="Supervised", cms_label=""):
    output_dir = os.path.join(dir_name, "postfit_bonly_stitched")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if tagger == "Supervised":
        skip_x_label = 2
    elif tagger == "Unsupervised":
        skip_x_label = 4

    for region in ["A", "B", "C", "D"]:
        payload = __collect_stitched_region_payload(
            histograms_by_category=histograms_by_category,
            region=region,
            bkg_key="fit_b_bkg",
            data_key="fit_b_data",
            sig_key_map={"Signal": "prefit_signal"},
            category_order=category_order,
            skip_x_label=skip_x_label,
        )
        if payload is None:
            continue

        outname = os.path.join(output_dir, f"plot_postfit_bonly_stitched_{region}_{year}")
        __draw_stitched_plot(
            payload=payload,
            outname=outname,
            year=year,
            fit_label=f"Region {region}, postfit bkg.-only",
            param_config=param_config,
            ratio_label="Data/Bkg.",
            draw_sb_line=False,
            tagger=tagger,
            cms_label=cms_label,
        )


def save_postfit_sb_stitched_plots(histograms_by_category, dir_name, param_config, year, category_order=None, tagger="Supervised", cms_label=""):
    output_dir = os.path.join(dir_name, "postfit_sb_stitched")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if tagger == "Supervised":
            skip_x_label = 2
    elif tagger == "Unsupervised":
        skip_x_label = 4

    for region in ["A", "B", "C", "D"]:
        payload = __collect_stitched_region_payload(
            histograms_by_category=histograms_by_category,
            region=region,
            bkg_key="fit_s_bkg",
            data_key="fit_s_data",
            sig_key_map={
                "Signal": "prefit_signal",
                "Signal × μ": "fit_s_signal",
            },
            category_order=category_order,
            skip_x_label=skip_x_label,
        )
        if payload is None:
            continue

        outname = os.path.join(output_dir, f"plot_postfit_sb_stitched_{region}_{year}")
        __draw_stitched_plot(
            payload=payload,
            outname=outname,
            year=year,
            fit_label=f"Region {region}, postfit S+B",
            param_config=param_config,
            ratio_label="Data/Bkg.",
            draw_sb_line=True,
            tagger=tagger,
            cms_label=cms_label,
        )


def _cli_mapping(datacard):
    mapping = {}
    with open(datacard) as card:
        for line in card:
            if "shapes" not in line or "Bkg" not in line or "wspace_abcd_model" not in line:
                continue
            words = line.split()
            channel = next((word for word in words if word.startswith("ch")), None)
            obj = next((word.split(":", 1)[1] for word in words if "wspace_abcd_model" in word and ":" in word), None)
            if not channel or not obj:
                continue
            bits = obj.split("_")
            if len(bits) >= 4 and bits[0] == "bkg" and bits[-1] in ("A", "B", "C", "D"):
                mapping.setdefault(bits[1], []).append([channel, "", bits[-1]])
    for category, entries in list(mapping.items()):
        if "3PSVJ" in category:
            source = mapping.get(category.replace("3PSVJ", "2SVJ", 1), [])
            present = {entry[2] for entry in entries}
            entries.extend(entry for entry in source if entry[2] not in present)
    return mapping


def _cli_main():
    import argparse
    from types import SimpleNamespace
    parser = argparse.ArgumentParser(description="Standalone stitched prefit/postfit plotting script.")
    parser.add_argument("-i", "--input-files", nargs="+", required=True, help="FitDiagnostics ROOT file (one file).")
    parser.add_argument("-d", "--datacard", required=True)
    parser.add_argument("-w", "--workspace-file", required=True)
    parser.add_argument("-o", "--output-folder", required=True)
    parser.add_argument("-t", "--tagger", choices=("Supervised", "Unsupervised"), required=True)
    parser.add_argument("-y", "--year", required=True)
    parser.add_argument("--signal-point", default="mMed2000_mDark20_rinv0p3_yukawa1")
    parser.add_argument("--cms-label", default="")
    args = parser.parse_args()
    if len(args.input_files) != 1:
        parser.error("--input-files currently accepts exactly one FitDiagnostics ROOT file")
    mapping = _cli_mapping(args.datacard)
    if not mapping:
        raise RuntimeError("No ABCD channels found in the datacard")
    fits = check_fit_results(args.input_files[0])
    abcd = SimpleNamespace(remap_back_postfit=False, map_regions={region: region for region in ("A", "B", "C", "D")})
    summary = {}
    for category in mapping:
        yields = __read_data_yields(args.workspace_file, False, args.signal_point, mapping, category)
        histograms = {}
        for fit in fits:
            histograms[fit + "_bkg"] = __get_bkg_histograms(args.input_files[0], fit, mapping, category, abcd)
            histograms[fit + "_data"] = __get_data_histograms(args.input_files[0], yields, fit, mapping, category, abcd)
            if fit != "fit_b":
                histograms[fit + "_signal"] = __get_signal_histograms(args.input_files[0], fit, mapping, category, abcd)
        base, split_year = (category.split("Y", 1) + [args.year])[:2] if "Y" in category else (category, args.year)
        summary.setdefault(base, {})[split_year] = histograms
    os.makedirs(args.output_folder, exist_ok=True)
    order = [cat for cat in ("0SVJ", "1SVJ", "2SVJ", "3PSVJ") if cat in summary]
    combined = {}
    for category, years in summary.items():
        combined[category] = {}
        for fit in fits:
            for flavor in ("bkg", "data", "signal"):
                if fit + "_" + flavor in next(iter(years.values())):
                    combined[category][fit + "_" + flavor] = __combine_histograms_years(years, fit, flavor)
    save_prefit_stitched_plots(combined, args.output_folder, args.signal_point, args.year, order, args.tagger, args.cms_label)
    if "fit_b" in fits:
        save_postfit_b_stitched_plots(combined, args.output_folder, args.signal_point, args.year, order, args.tagger, args.cms_label)
    if "fit_s" in fits:
        save_postfit_sb_stitched_plots(combined, args.output_folder, args.signal_point, args.year, order, args.tagger, args.cms_label)


if __name__ == "__main__":
    _cli_main()
