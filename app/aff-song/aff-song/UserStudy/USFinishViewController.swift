//
//  USFinishViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class USFinishViewController : AffUIViewController {
        
    @IBAction func FinishButtonClicked(_ sender: Any) {
        self.navigationController?.dismiss(animated: true, completion: nil)
    }
}
