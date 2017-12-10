//
//  AppDelegate.swift
//  aff-song
//
//  Created by Charlie Hewitt on 08/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        refreshAuth()
        return true
    }

    func applicationWillResignActive(_ application: UIApplication) {
        // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
        // Use this method to pause ongoing tasks, disable timers, and invalidate graphics rendering callbacks. Games should use this method to pause the game.
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
        // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
    }

    func applicationWillEnterForeground(_ application: UIApplication) {
        // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    }

    func applicationWillTerminate(_ application: UIApplication) {
        // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
    }
    
    func refreshAuth() {
        // get a new auth key from spotify
        let url: String = "https://accounts.spotify.com/api/token"
        var request: URLRequest = URLRequest(url: URL(string: url)!)
        let bodyData = "grant_type=refresh_token&client_id=75f91608ce154091a1f419be415cbdda&client_secret=a6609b179eb140b086c2f6cc2c35adf4&refresh_token=AQBeDOqXW6kerokqh6WbEexkOQH5FtWGfDvw2DLePMofXTAVOUGsygF5iWYx0jtCjBUFO31qslCGNcRNh6vXGv9wxxbEuyNyWFy-t1YuYON2UD_ySlzSCcjAWsI2YiKzFAc"
        request.httpBody = bodyData.data(using: String.Encoding.utf8);
        request.httpMethod = "POST"
        
        let session = URLSession.shared
        session.dataTask(with: request) {data, response, err in
            do {
                let json = try JSONSerialization.jsonObject(with: data!, options: .allowFragments) as! [String:Any]
                if  let authKey = json["access_token"] {
                    UserDefaults.standard.set(authKey, forKey: "SpotifyAuthToken")
                }
            } catch let error {
                print(error.localizedDescription)
            }
            }.resume()
    }
}

