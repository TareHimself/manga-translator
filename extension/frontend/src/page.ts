
// Windows to allow right click in
if(["https://weloma."].some( c => window.location.origin.startsWith(c))){
    const oldPreventDefault = MouseEvent.prototype.preventDefault
    MouseEvent.prototype.preventDefault = function (this: MouseEvent){
        if(this.type !== "contextmenu"){
            oldPreventDefault.bind(this)()
        }
    }
}