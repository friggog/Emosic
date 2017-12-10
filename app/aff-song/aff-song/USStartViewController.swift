//
//  USStartViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class USStartViewController : AffUIViewController {
    
    @IBAction func cancelButtonClicked(_ sender: Any) {
        navigationController?.dismiss(animated: true, completion: nil)
    }
}
