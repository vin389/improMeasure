#############################################################################
# $Id: menus.tcl,v 1.3 2012/01/22 03:13:43 rozen Exp rozen $
#
# menus.tcl procedures to edit application menus
#
# Copyright (C) 2000 Christian Gavin
#
# This program is free software; you can redistribute it and/or
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
##############################################################################

# Rozen. This seems stuff for user definition of menus.

# I had a bit of an experimental hassle to get it to work at all. I
# found that to make a menu for the top level window one needed to add
# not commands to the menu but cascades then one could fill out those
# cascades like File, Edit, View, etc. with commands or cascaded
# menues.  Damn lack of documentation.

# Contains most of the menu editor routines. Also involved are
# routines in misc.tcl. There is a whole section labeled Attributes
# editing in misc.tcl.

# When we start the menu edit it starts with vTcl:edit_target_menu

# When we click on an item in the listbox of the menu editor we jump
# to ::menu_edit::click_listbox,  in this module.
# ::menu_edit::show_menu about line 560 - where we start when a line
# in the editor is selected

# To aid in debugging
# {0 x y d v}
# vTcl:debug_dump_menu {target}

# fillProperties                    about line 700
# initBoxes                         about 567
# configCmd
# vTclWindow.vTclMenuEdit
# new_item                          where we add a new item.
# set_menu_item_defaults
# ::menu_edit::update_current_a
# ::menu_edit::click_listbox        where we go when selecting item in left side
# ::menu_edit::show_menu            Main routine for loading right hand side.
# keyRelease
# ::menu_edit::move_item

# nameapace ::vTcl::ui::attributes in misc.tcl. The ::ui:: stuff is in misc.tcl.

# The right hand side is filled in by ::menu_edit::initProperties" in
# vTclWindow.vTclMenuEdit
# The left hand side is filled in by ::menu_edit::fill_menu_list. Same place.

# Look at discoverOptions which I think runs much too often.

namespace eval ::menu_edit {

    # this contains a list of currently open menu editors
    variable menu_edit_windows ""

    proc {::menu_edit::delete_item} {top} {
        global widget
        set ::${top}::current_menu ""
        set ::${top}::current_index ""

        # this proc deletes the currently selected menu item

        set indices [$top.MenuListbox curselection]
        set index [lindex $indices 0]
        if {$index == ""} return
        set content [$top.MenuListbox get $index]
        set content [string trim $content]
        if {$content eq "<Menu>"} {
            set msg "Do not delete <menu> item.\nDelete parent instead."
            set choice [tk_dialog .foo "ERROR" $msg  question 0 "OK"]
            return
        }
       set listboxitems [set ::${top}::listboxitems]

        set reference [lindex $listboxitems $index]

        set menu [lindex $reference end-1]
        set pos  [lindex $reference end]

        if {$pos != -1} {
            set mtype [$menu type $pos]
            if {$mtype == "cascade"} {
                set submenu [$menu entrycget $pos -menu]
                ::menu_edit::delete_menu_recurse $submenu
            }

            $menu delete $pos
            vTcl:init_wtree
        }

        ::menu_edit::fill_menu_list $top [set ::${top}::menu]
        click_listbox $top
        vTcl:change
    }

    proc {::menu_edit::delete_menu_recurse} {menu} {
        global widget vTcl
        # unregister
        catch {namespace delete ::widgets::$menu} error

        set last [$menu index end]
        while {$last != "none"} {

            set mtype [$menu type $last]

            if {$mtype == "cascade"} {
                set submenu [$menu entrycget $last -menu]
                ::menu_edit::delete_menu_recurse $submenu
            }

            if {$mtype == "tearoff"} {
                $menu configure -tearoff 0
            } else {
                $menu delete $last
            }

            set last [$menu index end]
        }

        destroy $menu
    }

    proc {::menu_edit::enable_editor} {top enable} {

        global widget

        set ctrls "$top.NewMenu      $top.DeleteMenu
                   $top.MoveMenuUp   $top.MoveMenuDown"

        switch $enable {
            0 - false - no {
               set ::${top}::backup_bindings [bindtags $widget($top,MenuListbox)]
                bindtags $widget($top,MenuListbox) dummy
                bindtags $widget($top,NewMenu)     dummy
                $top.MenuListbox configure -background gray

                foreach ctrl $ctrls {
                    set ::${top}::$ctrl.state      [$ctrl cget -state]
                    set ::${top}::$ctrl.background [$ctrl cget -background]

                    $ctrl configure -state disabled
                }
                enableProperties $top 0
            }
            1 - true - yes {
                foreach ctrl $ctrls {
                    if {[info exists ::${top}::$ctrl.state]} {
                        $ctrl configure -state  [set ::${top}::$ctrl.state] \
                            -background [set ::${top}::$ctrl.background]
                    }
                }

                if {[info exists ::${top}::backup_bindings]} {
                    bindtags $widget($top,MenuListbox) \
                        [set ::${top}::backup_bindings]
                }
                bindtags $widget($top,NewMenu) ""
                # $top.MenuListbox configure -background white
                $top.MenuListbox configure -background #d9d9d9
                enableProperties $top 1
                ::menu_edit::click_listbox $top
            }
        }
    }

    proc {::menu_edit::enable_all_editors} {enable} {

        variable menu_edit_windows

        set wnds $menu_edit_windows

        foreach wnd $wnds {
            ::menu_edit::enable_editor $wnd $enable
        }
    }

    proc {::menu_edit::set_uistate} {top} {
        foreach name [array names ::${top}::uistate] {
            if {$name != "Tearoff"} {
                $top.$name configure -state [set ::${top}::uistate($name)]
            }
        }
    }

    proc {::menu_edit::enable_toolbar_buttons} {top} {
        # The toolbar buttons are the buttons across the top of the
        # menu editor.
        set indices [$top.MenuListbox curselection]
		set size [$top.MenuListbox size]
		set index [lindex $indices 0]
        if {$index == "" || $index == 0} {
            array set ::${top}::uistate {
                DeleteMenu disabled  MoveMenuUp disabled MoveMenuDown disabled
                Tearoff disabled
            }
            ::menu_edit::set_uistate $top
            return
        }

        array set ::${top}::uistate { DeleteMenu normal Tearoff disabled }

        set m ""
		set i ""

        ::menu_edit::get_menu_index $top $index m i
        set j $i
        if {[$m cget -tearoff] == 1 && $i == 1} {
           set j [expr $i -1]
        }

        if {$j == 0} {
            array set ::${top}::uistate { MoveMenuUp disabled }
        } else {
            array set ::${top}::uistate { MoveMenuUp normal }
        }
        if {$index == [expr $size - 1]} {
            array set ::${top}::uistate { MoveMenuDown disabled }
        } else {
            array set ::${top}::uistate { MoveMenuDown normal }
        }		

             if {$i != -2 && [$m type $i] == "cascade"} {
            array set ::${top}::uistate { Tearoff normal }
        }

        ::menu_edit::set_uistate $top
    }

    proc {::menu_edit::fill_command} {top command} {
        # Recursive function which fills in the menu.
        global vTcl
        global widget
        ## if the command is in the form "vTcl:DoCmdOption target cmd",
        ## then extracts the command, otherwise use the command as is
        if {[regexp {vTcl:DoCmdOption [^ ]+ (.*)} $command matchAll realCmd]} {
            lassign $command dummy1 dummy2 command
        }

        $top.MenuText delete 0.0 end
        $top.MenuText insert end $command

        vTcl:syntax_color $widget($top,MenuText)
    }

    proc {::menu_edit::fill_menu} {top m {level 0} {path {}}} {
        # This procedure appears to rewrite the whole left side
        # whenever one adds a new item to a menu.
        global vTcl
        set size [$m index end]
        if {$path == ""} {
            set path $m
        } else {
            lappend path $m
        }
        if {$size == "none"} {return}
        for {set i 0} {$i <= $size} {incr i} {
            set mtype [$m type $i]
            if {$mtype == "tearoff"} continue

            lappend ::${top}::listboxitems [concat $path $i]

            set indent "    "

            for {set j 0} {$j < $level} {incr j} {
                append indent "    "
            }

            switch -exact $mtype {
                "cascade" {
                    set tearoff ""
                    set mlabel [$m entrycget $i -label]
                    set maccel [$m entrycget $i -accel]
                    set submenu [$m entrycget $i -menu]
#dpr submenu
# Problem is that the submenu may be of the form "$site_3_0.pop48" and site_x_x is not a global variable.
                    if {[regexp {\$(site.+)\.} $submenu trash clause]} {
#dmsg BINGO
#dpr clause
#dpr vTcl($clause)
                        set submenu $vTcl($clause)
#dpr submenu
                    }

                    # if {$submenu != ""} {
                    #     if {[$submenu cget -tearoff] == 1} {
                    #         set tearoff " ---X---"}
                    # }
                    if {$maccel != ""} {
                        $top.MenuListbox insert end \
                            "$indent${mlabel}   ($maccel)$tearoff"
                    } else {
                        # Adds line containing cascade entry label.
                        $top.MenuListbox insert end "$indent${mlabel}$tearoff"
                        # New stuff adding <Menu> entries.
                        append indent "    "
                        set mlabel "<Menu>"
                        lappend ::${top}::listboxitems [concat $submenu -2]
                        incr $i
                        $top.MenuListbox insert end "$indent${mlabel}$tearoff"
                    }
                    if {$submenu != ""} {
                        ::menu_edit::fill_menu \
                            $top $submenu [expr {$level + 1}] [concat $path $i]
                    }
                }
                "command" {
                    set mlabel   [$m entrycget $i -label]
                    set maccel   [$m entrycget $i -accel]
                    if {$maccel != ""} {
                        $top.MenuListbox insert end \
                            "$indent${mlabel}   ($maccel)"
                    } else {
                        $top.MenuListbox insert end \
                            "$indent${mlabel}"
                    }
                }
                "separator" {
                    $top.MenuListbox insert end "$indent<separator>"
                }
                "radiobutton" -
                "checkbutton" {
                    set mlabel [$m entrycget $i -label]
                    set maccel [$m entrycget $i -accel]
                    if {$mtype == "radiobutton"} {
                        set prefix "o "
                    } else {
                        set prefix "x "}
                    if {$maccel != ""} {
                        $top.MenuListbox insert end \
                            "$indent$prefix${mlabel}   ($maccel)"
                    } else {
                        $top.MenuListbox insert end \
                            "$indent$prefix${mlabel}"
                    }
                }
            }
        }
    }

    proc {::menu_edit::fill_menu_list} {top m} {
        # This fills in the left side of the menu editor. Clears the
        # left side of Menu Editor, and fills in the whole thing. Adds
        # the <Menu> at the top and then calls
        # ::menu_edit::fill_menu. Called when the menu editor is
        # created near bottom of this module and as the starting point
        # whenever the menu is changed - new item, move, delete, etc.
        global widget
        # let's try to save the context
        set indices [$top.MenuListbox curselection]
        set index [lindex $indices 0]
        if {$index == ""} {
            set index [set ::${top}::listbox_index]
        }
        set yview [lindex [$top.MenuListbox yview] 0]
        set ::${top}::listboxitems ""
        $top.MenuListbox delete 0 end
        # Here we put in the first item <Menu> at the top of the
        # box. We give it index of -1 which has effects when we hit it
        # again
        lappend ::${top}::listboxitems [list $m -1]
        $top.MenuListbox insert end "<Menu>"
        ::menu_edit::fill_menu $top $m
        set ::${top}::menu $m

        if {$index != ""} {
            $top.MenuListbox selection clear 0 end
            $top.MenuListbox selection set $index
        }

        $top.MenuListbox yview moveto $yview
    }

    proc {::menu_edit::get_menu_index} {top index ref_m ref_i} {
        upvar $ref_m m
        upvar $ref_i i
        set reference [set ::${top}::listboxitems]
        set reference [lindex $reference $index]
        set m [lindex $reference end-1]
        set i [lindex $reference end]
    }

    proc {::menu_edit::move_item} {top direction} {
        set indices [$top.MenuListbox curselection]
        set index   [lindex $indices 0]
        if {$index == ""} return
        set m ""
        set i ""
        ::menu_edit::get_menu_index $top $index m i
        if {$i == -2} {
            ::vTcl::MessageBox -icon warning -type ok -title "Warning"  \
                -message "Cannot move a submenu entry."
            return
        }
        # what is the new index ?
        switch $direction {
            up {
                set new_i [expr $i - 1]
            }
            down {
                set new_i [expr $i + 1]
            }
        }
        # let's save the old menu
        set old_config [$m entryconfigure $i]
        set mtype      [$m type $i]
        set optval ""
        # build a list of option/value pairs
        foreach option $old_config {
            lappend optval [list [lindex $option 0] [lindex $option 4]]
        }
        # delete the old menu
        $m delete $i
        # insert menu at the new place
        eval $m insert $new_i $mtype [join $optval]
        ::menu_edit::fill_menu_list $top [set ::${top}::menu]
        # let's select the same menu at its new location
        set size [$top.MenuListbox index end]
        for {set ii 0} {$ii < $size} {incr ii} {
            set mm ""
            set mi ""
            ::menu_edit::get_menu_index $top $ii mm mi
            if {$mm == $m && $new_i == $mi} {
                $top.MenuListbox selection clear 0 end
                $top.MenuListbox selection set $ii
                $top.MenuListbox activate $ii
                ::menu_edit::show_menu $top $ii
                break
            }
        }
        vTcl:change
    }

    proc {::menu_edit::new_item} {top type} {
        global widget
        global vTcl
		set indices [$top.MenuListbox curselection]
        set index [lindex $indices 0]
        if {$index == ""} return
        # The idea here is to allow selection to made at the <Menu>
        # line. We test for the menu line and if positive we back up
        # one line.
        set content [$top.MenuListbox get $index]
        if {[string first "<Menu>" $content] > 0} {
			incr index -1
        }

        set listboxitems [set ::${top}::listboxitems]
        set reference [lindex $listboxitems $index]
        set menu [lindex $reference end-1]
        set pos  [lindex $reference end]
        if {$pos == -2} {
            ::vTcl::MessageBox -icon warning -type ok -title "Warning"  \
                -message "Cannot add item to a submenu entry."
            return
        }
        if {$pos != -1} {
            set mtype [$menu type $pos]
            if {$mtype == "cascade"} {
                set menu [$menu entrycget $pos -menu]
            }
        }
        # Added "-compound left" to appropriate types below because
        # that seemed more natural. 12/1/21
		switch $type {
            "cascade" {
                set nmenu [vTcl:new_widget_name menu $menu]
                set vTcl($nmenu,font) ""
                # menu $nmenu -tearoff 0 \
                #     -activebackground $vTcl(analog_color_m) \
                #     -activeforeground $vTcl(active_fg) \
                #     -foreground $vTcl(actual_gui_menu_fg)  \
                #     -background $vTcl(actual_gui_menu_bg) \
			   	#     -font $vTcl(actual_gui_font_menu_name)
				menu $nmenu -tearoff 0 
                vTcl:widget:register_widget $nmenu -tearoff
                vTcl:setup_vTcl:bind $nmenu
                $menu add $type -label "NewCascade" -menu $nmenu \
                    -font $vTcl(actual_gui_font_menu_name) \
                    -compound left
                    #-activebackground $vTcl(analog_color_m) \
					-activeforeground $vTcl(active_fg) \
                    -foreground $vTcl(actual_gui_menu_fg)  \
                    -background $vTcl(actual_gui_menu_bg) \
                    -font $vTcl(actual_gui_font_menu_name)
                    ;# NEEDS WORK take out the default attributes above
                vTcl:init_wtree
                vTcl:active_widget $nmenu
                #vTcl::widgets::saveSubOptions $menu -label -menu \
                #          -activebackground -activeforeground  -background \
                #          -foreground -font -compound
                # The item does not yet show up in the menu editor.
            }
            "command" {
                $menu add $type -label "NewCommand"  \
                    -command "#" \
                    -font $vTcl(actual_gui_font_menu_name) \
                    -compound left
                # The command below lists the options we want to save.
                # Since I have put in the special options I want to make
                # sure that they are saved. Rozen
				#vTcl::widgets::saveSubOptions $menu -label -command -font\
                    -foreground -background -activeforeground -activebackground \
                    -menu -compound
                #::menu_edit::set_menu_item_defaults $menu last   ;# Rozen
            }
            "radiobutton" {
                #set toplevel [findToplevel $menu]
                $menu add $type -label "NewRadio"  \
                    -command "#" \
                    -variable menuSelectedButton \
                    -font $vTcl(actual_gui_font_menu_name) \
                    -compound left
                    # -variable ::${toplevel}::menuSelectedButton -compound left
                # The default value of the option -value is value of
                # -label, in this case "NewRadio". See the tcl menu man page.
                #vTcl::widgets::saveSubOptions $menu -label -command -variable \
                    -font -foreground -background -activeforeground \
                    -activebackground -tearoff -compound
                # set_menu_item_defaults
                #::menu_edit::set_menu_item_defaults $menu last   ;# Rozen
            }
            "checkbutton" {
                set num 1
                #set toplevel [findToplevel $menu]
                # while {[info exists ::${toplevel}::check${num}]} {incr num}
                while {[info exists vTcl(check${num})]} {incr num}
                set vTcl(check${num}) checks_$num
                $menu add $type -label "NewCheck"  \
                    -command "#" \
                    -variable $vTcl(check${num}) \
                    -font $vTcl(actual_gui_font_menu_name) \
                    -compound left
                    # -variable ::${toplevel}::check${num}  -compound left
                 # set ::${toplevel}::check${num} 1
                 vTcl::widgets::saveSubOptions $menu -label -command -variable \
                        -font -foreground -background -activeforeground \
                     -activebackground -compound
            }
            "separator" {
                $menu add $type
                vTcl::widgets::saveSubOptions $menu -background -foreground
                # Line below because separator has no attributes of
                # font, active forground, active background or even
				# forground.
                #vTcl:prop:default_opt $menu -background vTcl(w,opt,-background)\
                        $vTcl(actual_gui_menu_bg) last 1
            }
        }
        #vTcl:change
        ::menu_edit::fill_menu_list $top [set ::${top}::menu]
        # At this point the new item does show up in menu editor.
	}

    proc set_menu_item_defaults {menu {index {}}} {
        # Added by Rozen.  Since I have added preferences for
        # background foreground and fonts for menus I have to pass not
        # the standard defaults but the ones that I have defined.
        # vTcl:prop:default_opt is in propmgr.tcl.
        global vTcl
        foreach def {-activebackground -activeforeground
                     -background -foreground -font} {
            switch -exact -- $def {
                -background {
                    vTcl:prop:default_opt $menu $def vTcl(w,opt,$def) \
                        $vTcl(actual_gui_menu_bg) $index 1
                }
                -foreground {
                    vTcl:prop:default_opt $menu $def vTcl(w,opt,$def)\
                        $vTcl(actual_gui_menu_fg) $index 1
                }
                -font {
                    vTcl:prop:default_opt $menu $def vTcl(w,opt,$def) \
                        $vTcl(actual_gui_font_menu_desc) $index 1
                }
                -activebackground {
                    vTcl:prop:default_opt $menu $def vTcl(w,opt,$def) \
                        $vTcl(actual_gui_menu_active_bg) $index 1
                }
                -activeforeground {
                    vTcl:prop:default_opt $menu $def vTcl(w,opt,$def) \
                        $vTcl(actual_gui_menu_active_fg) $index 1
                }
            }
        }
    }

    proc findToplevel {menu} {
        set toplevel [winfo toplevel $menu]
        while {[winfo class $toplevel] != "Toplevel"} {
            set toplevel [winfo parent $toplevel]}
        return $toplevel
    }

    proc {::menu_edit::post_context_new_menu} {top X Y} {
        global widget
        tk_popup $widget($top,NewMenuContextPopup) $X $Y
    }

    proc {::menu_edit::post_new_menu} {top} {
        global widget
        set x [winfo rootx  $widget($top,NewMenu)]
        set y [winfo rooty  $widget($top,NewMenu)]
        set h [winfo height $widget($top,NewMenu)]

        tk_popup $widget($top,NewMenuToolbarPopup)  $x [expr $y + $h]
    }

    ## get initial values for the checkboxes
    proc initBoxes {top m} {
        variable target
        set nm ::widgets::${m}::subOptions
        namespace eval $nm {}
        ## first time ? if so, check values that are not the default
        if {![info exists ${nm}::save]} {
            ## how many subitems ?
            set size [$m index end]
            if {$size == "none"} {
                return
            }
            for {set i 0} {$i <= $size} {incr i} {
                set conf [$m entryconfigure $i]
                foreach opt $conf {
                    set option  [lindex $opt 0]
                    set default [lindex $opt 3]
                    set value   [lindex $opt 4]
                    if {$value != $default} {
                        set ${nm}::save($option) 1
                    }
                }
            }
        }
    }

    proc {::menu_edit::show_menu} {top index} {
        global vTcl
        global widget
		set name ::${top}::listboxitems
        eval set reference $\{$name\}
        set reference [lindex $reference $index]
        set m [lindex $reference end-1]
		set i [lindex $reference end]
        ###########################################################
        # Beginning of coloring the listbox on the right hand side and
        # recoloring the last selection.
        set index $vTcl(listbox_selection)
        set lb $top.cpd24.01.cpd25.01
        # $lb itemconfigure $index -background blue -foreground white
        $lb itemconfigure $index -foreground black -background #d9d9d9
		# Update color of previous selection.
        catch {[$lb itemconfigure $vTcl(last_colored) \
                   -foreground black -background #d9d9d9]}
        set vTcl(last_colored) $index
        update
        # End of coloring code.
        ###########################################################
        if {$i == -1} {
            enableProperties $top 0       ;# Turn all the property fields off.
            set ::${top}::current_menu  $m
			set ::${top}::current_index $i
            ::menu_edit::enable_toolbar_buttons $top
            fillProperties $top $m -1         ;# Rozen Needs Work.
            return
        }
        if {$i == -2} {
            enableProperties $top 0       ;# Turn all the property fields off.
            set ::${top}::current_menu  $m
            set ::${top}::current_index $i
            ::menu_edit::enable_toolbar_buttons $top
            fillProperties $top $m -1         ;# Rozen Needs Work.
            return
        }

        #set mtype  [$m type $i]         ;#  NEEDS WORK Doesn't seem to be used!

        set ::${top}::current_menu  $m
   set ::${top}::current_index $i  ;# NEEDS WORK Really concerned about change!
        set ::${top}::current_index $index
        set ::${top}::listbox_index $index
        initBoxes $top $m
        fillProperties $top $m $i
        ::menu_edit::enable_toolbar_buttons $top
    }

    proc setGetBox {top option {var {}}} {
		set m [set ::${top}::current_menu]
        set i [set ::${top}::current_index]
        if {$m == "" || $i == ""} {return}

        set nm ::widgets::${m}::subOptions
        if {$var != ""} {
            set ${nm}::save($option) [set $var]
        } else {
            if {[info exists ${nm}::save($option)]} {
                return [set ${nm}::save($option)]
            } else {
                return 0
            }
        }
        vTcl:change
    }


	proc correct_menu_index { top j } {
		# Correct the jth value of the MenuListbox to the value of the
		# menu entry.

		# get a list of MenuListbox entries between 0 and j.
		set entries [$top.MenuListbox get 0 $j]
        if {$j == 1} { return 1 }
		set count -1
		foreach e $entries {
			if {[regexp {<Menu>} $e]} {
				set count -2
			} elseif {[regexp {<Menu>} $e]} {
				incr count -1
			}
			incr count
		}
		return $count    ;#[expr {$j - $count - 1}]
	}
		
    proc keyRelease {top option var boxvar} {
        # I think that all this does is check to see if option is the
        # default value
        global vTcl
return
		set m [set ::${top}::current_menu]
        set i [set ::${top}::current_index]
		# set i [::menu::correct_menu_index $i]
		#set i [correct_menu_index $top $i]
		
		#set selection_index [$top.MenuListbox curselection]
		#set selection_index $vTcl(listbox_selection)
		set selection_index $i
        # set selection_value [$top.MenuListbox get $selection_index]
        set selection_value [$top.MenuListbox get $i]		
        if {$m == "" || $i == ""} {return}
        if {$i == -1} return        ;# Rozen Again, allow change to <Menu>
        if {$i == -2} return        ;# Rozen Again, allow change to <Menu>
        if {[regexp {<Menu>} $selection_value]} {
            set j [expr {$i - 1}]
            foreach item ::${top}::listboxitems {
                if {[lindex $item 1] == $j} {
                    set m [lindex $item 0]
                    break
                }
            }
            set conf [$m cget $option]
        } else {
			set i [correct_menu_index $top $i]
			set cc [$m entryconfigure $i]
			set mc [$m configure]
			#if {$i < 0} {set i 0}
			#set i 1 ;# 1 fails; 2 works
			set conf [$m entryconfigure $i $option]
        }
        set default [lindex $conf 3]
        set value [set $var]
        if {[info exists vTcl(att_changed)] && $vTcl(att_changed)} {
            set vTcl(att_changed,$m,$option) 1
        }
        if {$value != $default} {
          set $boxvar 1
            setGetBox $top $option $boxvar
        }
    }

    proc discoverOptions {} {
        # Purpose is to get a list of all options associated with
        # menus. It does that by building a menu, add one entry for
        # each legal entry type and then does an entry configure of
        # that entry appending the result to the entry list.

		# This is ripe for change - run every time with a big switch
		# and new parameter to return only options for the one entry
		# type. It would have to be called from a different place and
		# with every selection.
		
        menu .m -tearoff 0
        .m add cascade -menu ".m.cascade" -label "test"
        menu .m.cascade -tearoff 0
        set options [.m entryconfigure 0]

        .m.cascade add command -label "command"
        set options [concat $options [.m.cascade entryconfigure 0]]

        .m.cascade add radiobutton -label "command"
        set options [concat $options [.m.cascade entryconfigure 1]]

        .m.cascade add checkbutton -label "command"
        set options [concat $options [.m.cascade entryconfigure 2]]

        .m.cascade add separator
        set options [concat $options [.m.cascade entryconfigure 3]]
        # We now have the option list so get rid of the menu
        destroy .m.cascade
        destroy .m
        # Each option is a list. Want only first element of the list.
        set optionsList ""
        foreach option $options {
            lappend optionsList [lindex $option 0]
        }
        return [lsort -unique $optionsList]
    }

    proc checkAttribute {m option checkVar} {
        set nm ::widgets::${m}::subOptions
        if {[info exists ${nm}::save($option)]} {
            set $checkVar [set ${nm}::save($option)]
        } else {
            set $checkVar 0
        }
    }

    proc update_default_properties {menu index option value} {
        # Rozen. This routine sets the value into the option field of
        # the ith element of the menu. It seems that when the menu
        # item is created the defaults and value are not set by the
        # interpreter.  I think that I need to do that in order to get
        # anything different from Tk default colors and font.
        if {$index > -1} {
            $menu entryconfigure $index $option $value
        }
    }

    # namespace  ::vTcl::ui::attributes  in misc.tcl

    proc fillProperties {top m i} {
        # This appears to turn on the ones which are appropriate for
        # the menu type. This is part of a chain which starts with
        # ::menu_edit::click_listbox which is entered when
        # <<ListboxSelect>> occurs in the left hand side of menu
        # editor.

        # Since I want menu items to show up with default colors and
        # fonts specified in the preferences, I had to hack this up
        # with several special cases. See below at the switch which I
        # use to tranfer default values into values actually to be
        # used.
        global vTcl
        # set selection_index [$top.MenuListbox curselection]
        # set selection_value [$top.MenuListbox get $selection_index]
        upvar ::${top}::enableData enableData
        upvar ::${top}::allOptions allOptions
        if {$i == -1} {   # So that we can change menu properties. Rozen
            set properties [$m configure]

        } else {
            set properties [$m entryconfigure $i]
        }
        set currentOptions ""
        set skip_comment 0
        foreach property $properties {
            set option [lindex $property 0]
            if {$option == "-menu"} {
                # I infer that this is a cascade entry.
                set skip_comment 1
                break
            }
        }
        foreach property $properties {
            set option [lindex $property 0]
            if {$skip_comment && $option == "-command"} {
                # Don't let user specify a command for a cascade entry.
                continue
            }
            if {$skip_comment && $option == "-menu"} {
                # Don't let user specify a command for a cascade entry.
                continue
            }
            set value  [lindex $property 4]
            if {$option eq "-command"} {
                # Remove leading "#" characters before displaying command value.
                regsub {^[#]+} $value "" value
            }
            set variable ::${top}::optionsValues($option)
            set $variable $value
            lappend currentOptions $option
        }
        foreach option $allOptions($top) {
			set variable ::${top}::optionsValues($option)
            set f [$top.MenuCanvas].f.$option
            # first uncheck the box, then check it if needed
            set checkVar [::vTcl::ui::attributes::getCheckVariable $f $option]
            set $checkVar 0
            # enable/disable option if it does/does-not apply to subitem
            if {[lsearch -exact $currentOptions $option] == -1} {
              ::vTcl::ui::attributes::enableAttribute $enableData($top,$option) 0
            } else {
              ::vTcl::ui::attributes::enableAttribute $enableData($top,$option) 1
                checkAttribute $m $option $checkVar
            }
        }

        ## make sure the -command option substitutes %widget and %top with the
        ## correct value
        set f [$top.MenuCanvas].f.-command
        ::vTcl::ui::attributes::setCommandTarget $f -command $m
    }

    proc enableProperties {top enable} {
        upvar ::${top}::enableData enableData
        upvar ::${top}::allOptions allOptions
        foreach option $allOptions($top) {
        ::vTcl::ui::attributes::enableAttribute $enableData($top,$option) $enable
        }
    }

    proc enable_select_properties {top} {
        # To turn on several of the properties of the menu as a whole.
        upvar ::${top}::enableData enableData
        foreach option [list -background -foreground -font -activebackground \
                   -activeforeground -command] {
           ::vTcl::ui::attributes::enableAttribute $enableData($top,$option) 1
        }
    }

	proc configCmd {top option variable} {
		global vTcl
		set m [set ::${top}::current_menu]
		set i [set ::${top}::current_index]
		set selection_value [$top.MenuListbox get $i]
		# The latest problem is to deterime whether the menu is a menu
		# bar or a popup menu.
		if {[string first pop $m] > -1} {
			set type popup
		} else {
			set type menubar
		}
		if {$m == "" || $i == ""} {return}
		# if {$i == -1} {  # Rozen to allow setting of <Menu> options.}
		if {$i < 0} {  # Rozen to allow setting of <Menu> options.
			if {$option eq "-font" && [set $variable] eq ""} {
				::vTcl::MessageBox -title Error -message \
					"Blank value for option \"$option\" not allowed."
				return
			}
			$m configure $option [set $variable]
			return
		} elseif {[regexp {<Menu>} $selection_value]} {
			set j [expr {$i - 1}]
			foreach item ::${top}::listboxitems {
				if {[lindex $item 1] == $j} {
					set m [lindex $item 0]
					break
				}
			}
			set properties [$m configure]
		} else {
			set v [set $variable]
			# This is how I can put anything I want into the command
			# field! For now I am making the command a simple comment so
			# that hitting the command does nothing. Rozen
			if {$option == "-command"} {
				set v "#$v"
			}		
			set listboxitems [set ::${top}::listboxitems]
			set start_reference [lindex $listboxitems $i]
			set start_menu [lindex $start_reference end-1]
			set j 0
			set l 0
			if {$type eq "menubar"} {		
				while {$l <= $i} {
					set reference [lindex $listboxitems $l]
					set menu [lindex $reference end-1]
					set selection_v [$top.MenuListbox get $l]
					if {$menu == $start_menu} {
						if {[regexp {<Menu>} $selection_v]} {
							set j -1
						} else {
							incr j
						}						
					}
					incr l
				}
			}
			if {$type eq "popup"} {
				while {$l <= $i} {
					set reference [lindex $listboxitems $l]
					set menu [lindex $reference end-1]
					set selection_v [$top.MenuListbox get $l]
					if {$menu == $start_menu} {
						if {[regexp {<Menu>} $selection_v]} {
							# Determine if the menu is a cascade by
							# seeing if the <Menu> entry is indented,
							# i.e. if position is greater than zero,
							set position \
								[string first "<Menu>" $selection_v]
							if {$position == 0} {
								set j 0
							} else {
								set j -1
							}				   
						} else {
							incr j
						}
					}
					incr l
				}
			}
			set k $j
			# Catch - in the case where the option value is incomplete
			# causing an illegal entryconfigure, like "-" for
			# underline option.
			catch {$m entryconfigure $k $option $v}
			if {$option == "-label" || $option == "-accelerator"} {
				::menu_edit::fill_menu_list $top [set ::${top}::menu]
			}
		}
	}

    proc initProperties {top m} {
		# This is called as ::menu_edit::initProperties above;
        # This works differently from the Attribute Editor which shows
        # only the attributes appropriate to the selected widget. Here
        # we show them all but only the appropriate ones are active.

        # When the menu editor is started this routine fills in
        # the right hand column with the list of properties but they
        # are not enabled.

        # This sets the various options to set an array entry to
        # receive the option value.  See "set variable ..." below.
        upvar ::${top}::enableData enableData
        upvar ::${top}::allOptions allOptions
        set options [discoverOptions]
        set allOptions($top) $options
        set target($top) $m
        foreach option $options {
            set variable ::${top}::optionsValues($option)
            set $variable ""
            set f [$top.MenuCanvas].f.$option
            if {[winfo exists $f]} {destroy $f}
            frame $f
            set config_cmd "::menu_edit::configCmd $top $option $variable"
            # the newAttribute routine is located in misc.tcl under
            # attribute editing about 2/3 of the way down under
            # ::vTcl::ui::attributes namespace.
            set enableData($top,$option) \
                [::vTcl::ui::attributes::newAttribute  $target($top) $f \
                 $option $variable $config_cmd  \
                 "::menu_edit::setGetBox $top"  \
                 "::menu_edit::keyRelease $top"]
            pack $f -side top -fill x -expand 0
            ::vTcl::ui::attributes::enableAttribute $enableData($top,$option) 0
        }
        ## calculate the scrolling region
        update idletasks
        set w [winfo width  [$top.MenuCanvas].f]
        set h [winfo height [$top.MenuCanvas].f]
        ${top}.MenuCanvas configure -scrollregion [list 0 0 $w $h]
    }

    proc {::menu_edit::toggle_tearoff} {top} {
        set indices [$top.MenuListbox curselection]
        set index   [lindex $indices 0]

        if {$index == ""} return

        set m ""
        set i ""

        ::menu_edit::get_menu_index $top $index m i

        set mtype [$m type $i]
        if {$mtype != "cascade"} return

        set submenu [$m entrycget $i -menu]
        if {$submenu == ""} return

        set tearoff [$submenu cget -tearoff]
        set tearoff [expr 1-$tearoff]
        $submenu configure -tearoff $tearoff

        ::menu_edit::fill_menu_list $top [set ::${top}::menu]
        ::menu_edit::show_menu $top $index
    }

    proc {::menu_edit::update_current} {top} {
        ::vTcl::ui::attributes::setPending  ;# Bottom of misc.tcl.
    }

    proc {::menu_edit::update_current_a} {top} {
        # Rozen. Put this one so that it will only be called only when
        # I exit the menu editor.  That way I can save the geometry of
        # the editor window as a preference.
        global vTcl
        ::vTcl::ui::attributes::setPending  ;# Bottom of misc.tcl.
        set geom [wm geometry $top]
        set vTcl(geometry,menu_editor) $geom
        vTcl:change
    }

    proc {::menu_edit::close_all_editors} {} {
        variable menu_edit_windows
        set wnds $menu_edit_windows
        foreach wnd $wnds {
            destroy $wnd
        }
        set menu_edit_windows ""
    }

    proc {::menu_edit::browse_image} {top} {
        set image [set ::${top}::entry_image]
        set r [vTcl:prompt_user_image2 $image]
        set ::${top}::entry_image $r
    }

    proc {::menu_edit::browse_font} {top} {
        set font [set ::${top}::entry_font]
        set r [vTcl:font:prompt_noborder_fontlist $font]

        if {$r != ""} {
            set ::${top}::entry_font $r
        }
    }

    proc {::menu_edit::click_listbox} {top} {
        # This is where we start when the listbox is clicked,
        # i.e. select a component in the listbox.  Then off to
        # menu_edit::show_menu
        global vTcl
        ::menu_edit::update_current $top
        set indices [$top.MenuListbox curselection]
        set index [lindex $indices 0]
        set vTcl(listbox_selection) $index
        if {$index != ""} {
			# Recolor tree on the left as well as showing new selection.
            ::menu_edit::show_menu $top $index
        }
    }

    proc {::menu_edit::ask_delete_menu} {top} {

        if {[::vTcl::MessageBox -message {Delete menu item ?} \
                           -title {Menu Editor} -type yesno] == "yes"} {
            ::menu_edit::delete_item $top
        }
    }

    # proc {::menu_edit::includes_menu} {top m} {

    #     # is it the root menu?
    #     if {[set ::${top}::menu] == $m} {
    #         return 0}

    #     set size [$top.MenuListbox index end]

    #     for {set i 0} {$i < $size} {incr i} {

    #         set mm ""
    #         set mi ""

    #         ::menu_edit::get_menu_index $top $i mm mi

    #         if {$mm != "" && $mi != -1 &&
    #             [$mm type $mi] == "cascade" &&
    #             [$mm entrycget $mi -menu] == $m} then {
    #             return $i
    #         }
    #     }
    #     # oh well
    #     return -1
    # }

    ## check if the menu to edit is a submenu in an already open
    ## menu editor, and if so, open that menu editor
    proc {::menu_edit::kill_existing_editor} { } {
        variable menu_edit_windows
        foreach top $menu_edit_windows {
           destroy $top
        }
    }

    ## check if the menu to edit is a submenu in an already open
    ## menu editor, and if so, open that menu editor
    # proc {::menu_edit::open_existing_editor} {m} {
    #     # Never called because I want to kill then menu editor instead.
    #     # let's check each menu editor
    #     variable menu_edit_windows
    #     foreach top $menu_edit_windows {

    #         set index [::menu_edit::includes_menu $top $m]

    #         if {$index != -1} {
    #             Window show $top
    #             raise $top
    #             $top.MenuListbox selection clear 0 end
    #             $top.MenuListbox selection set $index
    #             $top.MenuListbox activate $index
    #             ::menu_edit::show_menu $top $index
    #             return 1
    #         }
    #     }

    #     return 0
    # }

    # proc {::menu_edit::is_open_existing_editor} {m} {
    #     # let's check each menu editor
    #     variable menu_edit_windows

    #     foreach top $menu_edit_windows {

    #         if {[::menu_edit::includes_menu $top $m] != -1} then {
    #             return $top
    #         }
    #     }

    #     return ""
    # }

    ## refreshes the menu editor
    # proc {::menu_edit::refreshes_existing_editor} {top} {

    #     ::menu_edit::fill_menu_list $top [set ::${top}::menu]

    #     $top.MenuListbox selection clear 0 end
    #     $top.MenuListbox selection set 0
    #     $top.MenuListbox activate 0
    #     ::menu_edit::show_menu $top 0
    # }

    ## finds the root menu of the given menu
    proc {::menu_edit::find_root_menu} {m} {

        # go up until we find something that is not a menu
        set parent $m
        set lastparent $m

        while {[vTcl:get_class $parent] == "Menu"} {
            set lastparent $parent

            set items [split $parent .]
            set parent [join [lrange $items 0 [expr [llength $items] - 2] ] . ]
        }
        return $lastparent
    }

} ; # namespace eval

proc vTclWindow.vTclMenuEdit {base menu} {
    ##################################
    # OPEN EXISTING EDITOR IF POSSIBLE
    ##################################
    # if {[::menu_edit::open_existing_editor $menu]} then {
    #     return }
    # Check for and destroy any existing menu editor window.
    ::menu_edit::kill_existing_editor
    # always open a menu editor with root menu
    set original_menu $menu
    set menu [::menu_edit::find_root_menu $menu]

    global widget vTcl

    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    namespace eval $base {
        variable listboxitems  ""
        variable current_menu  ""
        variable current_index ""
        variable listbox_index ""
        variable enableData
        variable allOptions
        array set enableData {}
        array set allOptions {}
    }
    ###################
    # DEFINING ALIASES
    ###################
    vTcl:DefineAlias $base.cpd24.01.cpd25.01 MenuListbox vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.cpd24.01.cpd25.01.m24 \
        NewMenuToolbarPopup vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.cpd24.01.cpd25.01.m25 \
        NewMenuContextPopup vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.fra21.but21 NewMenu vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.fra21.but22 DeleteMenu vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.fra21.but23 MoveMenuUp vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.fra21.but24 MoveMenuDown vTcl:WidgetProc $base 1
    vTcl:DefineAlias $base.fra21.but22 MenuOK vTcl:WidgetProc $base 1

    ###################
    # CREATING WIDGETS
    ###################
    toplevel $base -class Toplevel -menu $base.m22 \
		-background $vTcl(pr,bgcolor) 
    #if {[::colorDlg::dark_color $vTcl((pr,bgcolor))]} {
	#	$base configure -foreground white}	
    wm overrideredirect $base 0
    wm focusmodel $base passive
    wm geometry $base 550x450+323+138
    #wm geometry $base 700x450+600+300
    wm withdraw $base
    wm maxsize $base 1284 1010
    wm minsize $base 100 1
    wm resizable $base 1 1
    wm title $base "Menu Editor"
    wm transient $base .vTcl

    menu $base.m22 -tearoff 0 -relief flat \
		-foreground $vTcl(pr,fgcolor) -background $vTcl(pr,bgcolor) \
		-font vTcl(pr,font_dft)
    $base.m22 add cascade \
        -menu "$base.m22.men23" -label Insert
    $base.m22 add cascade \
        -menu "$base.m22.men24" -label Delete
    $base.m22 add cascade \
        -menu "$base.m22.men36" -label Move
    menu $base.m22.men23 -tearoff 0
    $base.m22.men23 add command \
        -command "::menu_edit::new_item $base command" \
        -label {New command}
    $base.m22.men23 add command \
        -command "::menu_edit::new_item $base cascade" \
        -label {New cascade}
    $base.m22.men23 add command \
        -command "::menu_edit::new_item $base separator" \
        -label {New separator}
    $base.m22.men23 add command \
        -command "::menu_edit::new_item $base radiobutton" \
        -label {New radio}
    $base.m22.men23 add command \
        -command "::menu_edit::new_item $base checkbutteon" \
        -label {New check}
    menu $base.m22.men24 -tearoff 0  \
        -postcommand "$base.m22.men24 entryconfigure 0 -state \
            \[set ::${base}::uistate(DeleteMenu)\]"
    $base.m22.men24 add command \
        -command "::menu_edit::ask_delete_menu $base" \
        -label {Delete selected item.}
    menu $base.m22.men36 -tearoff 0 \
        -postcommand "$base.m22.men36 entryconfigure 0 -state \
            \[set ::${base}::uistate(MoveMenuUp)\]
            $base.m22.men36 entryconfigure 1 -state \
            \[set ::${base}::uistate(MoveMenuDown)\]"
    $base.m22.men36 add command \
        -command "::menu_edit::move_item $base up" \
        -label Up
    $base.m22.men36 add command \
        -command "::menu_edit::move_item $base down" \
        -label Down

    frame $base.fra21 \
		-background $vTcl(pr,bgcolor) \
        -borderwidth 2 -height 75 -width 125
    # Button but32 is the OK button with the check.
    ::vTcl::OkButton $base.fra21.but32 \
		-foreground $vTcl(pr,fgcolor) -background $vTcl(pr,bgcolor) \
        -command \
         "::menu_edit::update_current_a $base;
          #wm withdraw $base;
          destroy $base;
          vTcl:change"
    vTcl:set_balloon $base.fra21.but32 "Commit menu and close."
    frame $base.cpd24 \
		-background $vTcl(pr,bgcolor) \
        -height 100 -width 200
    frame $base.cpd24.01 \
        -background $vTcl(pr,bgcolor)
    # Add button
    set image [vTcl:light_or_dark_image plus]    ;# NEEDS WORK dark
    vTcl:toolbar_label $base.fra21.but21 \
        -image $image -background $vTcl(pr,bgcolor)
        # -image $image 
    bind $base.fra21.but21 <ButtonPress-1> {
        ::menu_edit::post_new_menu [winfo toplevel %W]
    }
    # Delete button
    set image [vTcl:light_or_dark_image minus]  
    vTcl:toolbar_button  $base.fra21.but22 \
        -image $image -background $vTcl(pr,bgcolor) \
		-command "::menu_edit::ask_delete_menu $base"
	# Up button
    set image [vTcl:light_or_dark_image page_up]
    vTcl:toolbar_button  $base.fra21.but23 \
		-background $vTcl(pr,bgcolor) \
        -image $image \
        -command "::menu_edit::move_item $base up" -image $image
    # Down button
    set image [vTcl:light_or_dark_image page_down] 
    vTcl:toolbar_button $base.fra21.but24 \
		-background $vTcl(pr,bgcolor) \
		-image $image \
        -command "::menu_edit::move_item $base down" -image $image
    ScrolledWindow $base.cpd24.01.cpd25   ;# This is the left hand side.
    listbox $base.cpd24.01.cpd25.01 \
		-background $vTcl(area_bg) -foreground black \
		-selectbackground blue -selectforeground white \
		-font $vTcl(pr,gui_font_text) 
	# listbox above aliased to MenuListbox
    $base.cpd24.01.cpd25.01 configure ;# -background red
    set vTcl(menu,left_listbox) $base.cpd24.01.cpd25.01
    $base.cpd24.01.cpd25 setwidget $base.cpd24.01.cpd25.01

    bindtags $base.cpd24.01.cpd25.01 \
        "Listbox $base.cpd24.01.cpd25.01 $base all"
    bind $base.cpd24.01.cpd25.01 <Button-1> {
        focus %W
    }
    bind $base.cpd24.01.cpd25.01 <<ListboxSelect>> {
        # This is where we go when we click on a line in the listbox. Rozen
        ::menu_edit::click_listbox [winfo toplevel %W]
        after idle {focus %W}
    }
    bind $base.cpd24.01.cpd25.01 <ButtonRelease-3> {
		::menu_edit::update_current [winfo toplevel %W]
        set index [%W index @%x,%y]
        %W selection clear 0 end
        %W selection set $index
        %W activate $index
        if {$index != ""} {
            ::menu_edit::show_menu [winfo toplevel %W] $index
        }
        ::menu_edit::post_context_new_menu [winfo toplevel %W] %X %Y
    }
    menu $base.cpd24.01.cpd25.01.m24 \
		-background $vTcl(pr,bgcolor) -foreground $vTcl(pr,fgcolor) \
        -activeborderwidth 1 -tearoff 0
    $base.cpd24.01.cpd25.01.m24 add command \
        -accelerator {} -command "::menu_edit::new_item $base command" \
        -label {New command}
    $base.cpd24.01.cpd25.01.m24 add command \
        -accelerator {} -command "::menu_edit::new_item $base cascade" \
        -label {New cascade}
    $base.cpd24.01.cpd25.01.m24 add command \
        -accelerator {} -command "::menu_edit::new_item $base separator" \
        -label {New separator}
    $base.cpd24.01.cpd25.01.m24 add command \
        -accelerator {} \
        -command "::menu_edit::new_item $base radiobutton" \
        -label {New radio}
    $base.cpd24.01.cpd25.01.m24 add command \
        -accelerator {} \
        -command "::menu_edit::new_item $base checkbutton" \
        -label {New check}
    menu $base.cpd24.01.cpd25.01.m25 \
        -activeborderwidth 1 -tearoff 0 \
        -postcommand "$base.cpd24.01.cpd25.01.m25 entryconfigure 8 -state \
            \[set ::${base}::uistate(Tearoff)\]"
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} -command "::menu_edit::new_item $base command" \
        -label {New command}
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} -command "::menu_edit::new_item $base cascade" \
        -label {New cascade}
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} -command "::menu_edit::new_item $base separator" \
        -label {New separator}
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} \
        -command "::menu_edit::new_item $base radiobutton" \
        -label {New radio}
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} \
        -command "::menu_edit::new_item $base checkbutton" \
        -label {New check}
    $base.cpd24.01.cpd25.01.m25 add separator
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} \
        -command "::menu_edit::ask_delete_menu $base" \
        -label Delete...
    $base.cpd24.01.cpd25.01.m25 add separator
    $base.cpd24.01.cpd25.01.m25 add command \
        -accelerator {} -command "::menu_edit::toggle_tearoff $base" \
        -label Tearoff
    frame $base.cpd24.02

    # This is the window that has the attributes.
    ScrolledWindow $base.cpd24.02.scr84 -background $vTcl(pr,bgcolor)
    canvas $base.cpd24.02.scr84.can85 \
		-bg $vTcl(pr,bgcolor) \
        -highlightthickness 0 -yscrollincrement 20 -closeenough 1.0
    vTcl:DefineAlias "$base.cpd24.02.scr84.can85" "MenuCanvas"\
            vTcl:WidgetProc $base 1
    $base.cpd24.02.scr84 setwidget $base.cpd24.02.scr84.can85
    set vTcl(menu,right_canvas) $base.cpd24.02.scr84.can85
    frame $base.cpd24.03 \
        -background #ff0000 -borderwidth 2 -relief raised
    bind $base.cpd24.03 <B1-Motion> {
        set root [ split %W . ]
        set nb [ llength $root ]
        incr nb -1
        set root [ lreplace $root $nb $nb ]
        set root [ join $root . ]
        set width [ winfo width $root ].0

        set val [ expr (%X - [winfo rootx $root]) /$width ]

        if { $val >= 0 && $val <= 1.0 } {

            place $root.01 -relwidth $val
            place $root.03 -relx $val
            place $root.02 -relwidth [ expr 1.0 - $val ]
        }
    }
    ###################
    # SETTING GEOMETRY
    ###################
    pack $base.fra21 \
        -in $base -anchor center -expand 0 -fill x -side top
    pack $base.fra21.but32 \
        -in $base.fra21 -anchor center -expand 0 -fill none -side right
    pack $base.cpd24 \
        -in $base -anchor center -expand 1 -fill both -side top
    place $base.cpd24.01 \
        -x 0 -y 0 -width -1 -relwidth 0.6 -relheight 1 -anchor nw \
        -bordermode ignore
    pack $base.fra21.but21 \
        -in $base.fra21 -anchor center -expand 0 -fill none \
        -side left
    pack $base.fra21.but22 \
        -in $base.fra21 -anchor center -expand 0 -fill none \
        -side left
    pack $base.fra21.but23 \
        -in $base.fra21 -anchor center -expand 0 -fill none \
        -side left 
    pack $base.fra21.but24 \
        -in $base.fra21 -anchor center -expand 0 -fill none \
        -side left 
    pack $base.cpd24.01.cpd25 \
        -in $base.cpd24.01 -anchor center -expand 1 -fill both -side top
    #pack $base.cpd24.01.cpd25.01        Rozen BWidget
    place $base.cpd24.02 \
        -x 0 -relx 1 -y 0 -width -1 -relwidth 0.4 -relheight 1 -anchor ne \
        -bordermode ignore
    pack $base.cpd24.02.scr84 \
        -in $base.cpd24.02 -anchor center -expand 1 -fill both -side top
    place $base.cpd24.03 \
        -x 0 -relx 0.6 -y 0 -rely 0.9 -width 10 -height 10 -anchor s \
        -bordermode ignore

    pack [ttk::sizegrip $base.cpd24.sz -style "PyConsole.TSizegrip"] \
        -side right -anchor se

    vTcl:set_balloon $widget($base,NewMenu) \
        "Create a new menu item or a new submenu"
    vTcl:set_balloon $widget($base,DeleteMenu) \
        "Delete an existing menu item or submenu"
    vTcl:set_balloon $widget($base,MoveMenuUp) \
        "Move menu up"
    vTcl:set_balloon $widget($base,MoveMenuDown) \
        "Move menu down"

    frame [$base.MenuCanvas].f
    $base.MenuCanvas create window 0 0 -window [$base.MenuCanvas].f \
        -anchor nw -tag properties

    array set ::${base}::uistate {
        DeleteMenu disabled  MoveMenuUp disabled MoveMenuDown disabled
        Tearoff disabled
    }
    #############################
    # FILL IN MENU EDITOR `
    #############################
    # initializes menu editor
    ::menu_edit::initProperties $base $menu   ;# right hand side.
    ::menu_edit::fill_menu_list $base $menu   ;# left hand side.
    # keep a record of open menu editors
    lappend ::menu_edit::menu_edit_windows $base
    # initial selection
    # set initial_index [::menu_edit::includes_menu $base $original_menu]
    # if {$initial_index == -1} {
    #     set initial_index 0
    # }
    set initial_index 0
    $base.MenuListbox selection clear 0 end
    $base.MenuListbox selection set $initial_index
    $base.MenuListbox activate $initial_index
    ::menu_edit::click_listbox $base
    # when a menu editor is closed, should be removed from the list
    bind $base <Destroy> {
        set ::menu_edit::index \
            [lsearch -exact ${::menu_edit::menu_edit_windows} %W]
        if {${::menu_edit::index} != -1} {
            set ::menu_edit::menu_edit_windows \
                [lreplace ${::menu_edit::menu_edit_windows} \
                    ${::menu_edit::index} ${::menu_edit::index}]

            # clean up after ourselves
            namespace delete %W
        }
    }
    #######################
    # KEYBOARD ACCELERATORS
    #::menu_edit::
    ######################
    vTcl:setup_vTcl:bind $base
    # ok, let's add a special tag to override the <KeyPress-Delete> mechanism
    # I decided that I don't like this behavior, hence the block comment.
    bindtags $widget($base,MenuListbox) \
        "_vTclMenuDelete [bindtags $widget($base,MenuListbox)]"
    bind _vTclMenuDelete <KeyPress-Delete> {
        ::menu_edit::ask_delete_menu [winfo toplevel %W]
        # we stop processing here so that Delete does not get processed
        # by further binding tags, which would have the quite undesirable
        # effect of deleting the current toplevel...
        break
    }
    global vTcl
    # Following causes the editor window to be set from saved window
    set vTcl(menu_editor_base) $base  ;# Saves the base for later.
    # Moves the window to where we last had it.
    catch {wm geometry $base $vTcl(geometry,menu_editor)}
    # So the mouse wheel works inside the area rather than just over
    # the scroll bars.
    bind $vTcl(menu,right_canvas) <Enter> \
        {vTcl:bind_mousewheel  $::vTcl(menu,right_canvas)}
    bind $vTcl(menu,right_canvas) <Leave> \
        {vTcl:unbind_mousewheel $::vTcl(menu,right_canvas)}

    bind $vTcl(menu,left_listbox) <Enter> \
        {vTcl:bind_mousewheel  $::vTcl(menu,left_listbox)}
    bind $vTcl(menu,left_listbox) <Leave> \
        {vTcl:unbind_mousewheel $::vTcl(menu,left_listbox)}
    wm deiconify $base
}
