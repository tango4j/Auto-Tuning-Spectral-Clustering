text_yellow_info()
{
TEXT="=== [INFO] \e[93m$1 \e[39m" 
echo -e $TEXT
}

display_and_save_DER()
{
	der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' $1	)
	
	spk_err=$(grep "SPEAKER ERROR TIME =" $1 | grep -oP "\K[0-9]+([.][0-9]+)?" - | sed -n 2p)

	text_yellow_info "$2 Total Err. (DER) -[ $der % ] Speaker Err. [ $spk_err % ]"
}


