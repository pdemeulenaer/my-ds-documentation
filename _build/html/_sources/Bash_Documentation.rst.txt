======================
 Useful Bash commands
======================
      
Find 10 largest files in a directory:
----------------------


du -hsx * | sort -rh | head -10

Where,
du command -h option : display sizes in human readable format (e.g., 1K, 234M, 2G).
du command -s option : show only a total for each argument (summary).
du command -x option : skip directories on different file systems.
sort command -r option : reverse the result of comparisons.
sort command -h option : compare human readable numbers. This is GNU sort specific option only.
head command -10 OR -n 10 option : show the first 10 lines.




======================
 Useful GIT commands
======================

How to use git?
Once in team meeting Ovidijus talked about other way of using git. He has now set safety rules almost on all stash repositories which don’t allow us to push directly to master, rewrite master history and made changes without Pull Request.
Here are the different steps
1.            git checkout master <this selects master branch>
2.            git pull <this get newest data from master branch>
3.            git checkout –b “name” <this create new branch>
4.            change code
5.            git add <this needed to add changed files> (or git add --all in case we want to include all changes)
6.            git commit <after this command usually we shortly write that we changed>
7.            git push origin "name" <this push our branch "name" to remote (stash) and this allows to us create PR>
9.            create PR in stash <PR = Pull Request>
10.          merge PR in stash (ideally another team member actually does the merge)
Repeat these steps
 
Way to learn more about git and how it works: https://learngitbranching.js.org/


Screen commands
--------------------------

https://www.howtoforge.com/linux_screen
https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/



=====================
 Useful VIM commands
=====================

https://coderwall.com/p/adv71w/basic-vim-commands-for-getting-started 

Started learning to use Vim .. Some simple commands to get started
Vim has two mode .
1. Insert mode (Where you can just type like normal text editor. Press i for insert mode)
2. Command mode (Where you give commands to the editor to get things done . Press ESC for command mode)

Most of them below are in command mode

    x - to delete the unwanted character
    u - to undo the last the command and U to undo the whole line
    CTRL-R to redo
    A - to append text at the end
    :wq - to save and exit
    :q! - to trash all changes
    dw - move the cursor to the beginning of the word to delete that word
    2w - to move the cursor two words forward.
    3e - to move the cursor to the end of the third word forward.
    0 (zero) to move to the start of the line.
    d2w - which deletes 2 words .. number can be changed for deleting the number of consecutive words like d3w
    dd to delete the line and 2dd to delete to line .number can be changed for deleting the number of consecutive words

See much more in the link above.
