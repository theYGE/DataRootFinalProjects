import React from 'react'
import axios, { post } from 'axios';
import Button from 'react-bootstrap/Button';

class SimpleReactFileUpload extends React.Component {

    constructor(props) {
        super(props);
        this.state ={
            file:null,
            message: '',
            shouldShowFirst: false,
            shouldShowSecond: false,
            basic: ["", "", ""],
            trained: ["", "", ""],
            story: "",
            status: "Waiting for input"
        }
        this.onFormSubmit = this.onFormSubmit.bind(this)
        this.onChange = this.onChange.bind(this)
        this.onTextareaChange = this.onTextareaChange.bind(this)
        this.storiesUpload = this.storiesUpload.bind(this)
        this.onSecondFormSubmit = this.onSecondFormSubmit.bind(this)
    }
    onFormSubmit(e){
        console.log('Hi');
        e.preventDefault() // Stop form submit
        this.fileUpload(this.state.file).then((response)=>{
            // this.setState({
            //     message: 'Changed'
            // })
            console.log(response.data);
            // this.setState({
            //     message: response.data
            // })
            this.setState({
                message: response.data.label
            })
        })
    }
    onChange(e) {
        this.setState({file:e.target.files[0]})
    }

    onSecondFormSubmit(e){

        e.preventDefault() // Stop form submit
        this.storiesUpload(this.state.review).then((response)=>{
            // this.setState({
            //     message: 'Changed'
            // })
            console.log(response.data);
            // this.setState({
            //     message: response.data
            // })
            this.setState({
                story: response.data.story,
                basic: response.data.basic,
                trained: response.data.trained,
                status: "Recieved output"
            }, () => {
                console.log(this.state)
            })
        }).catch((error) => {
            console.log(error)
        })
    }

    storiesUpload(review){
        console.log("Submitting")
        const url = 'http://127.0.0.1:5000/second/predict';
        const formData = new FormData();
        formData.append('first_story',this.state.first_story)
        formData.append('second_story',this.state.second_story)
        formData.append('third_story',this.state.third_story)

        this.setState({
            status: "Loading"
        })

        return  post(url, formData)
    }

    onTextareaChange(e) {
        // this.setState({file:e.target.files[0]})
        console.log(e.target.value)
        this.setState({
            [e.target.id]: e.target.value
        }, ()=> {
            console.log(this.state)
        })
    }



    render() {
        return (

            <div align="center">
                <h1>DataRoot University Final Project</h1>
                <h4>Note: Stories may take a lot time to be generated since backend is running on EC2 with a CPU</h4>
                <br/>
                <h3><b>Status: {this.state.status}</b></h3>
                <br/>
                    <div>
                        <form onSubmit={this.onSecondFormSubmit}>
                            <div class = "row">
                                <h4 className="col-xl-12">Input</h4>
                                <label className="form-text col-xl-4">First story</label>
                                <label className="form-text col-xl-4">Second story</label>
                                <label className="form-text col-xl-4">Third story</label>
                            </div>
                            <div class="row">
                                <textarea className="form-control col-xl-4"
                                          id="first_story"
                                          onChange={this.onTextareaChange}
                                          rows="10">
                                </textarea>
                                <textarea className="form-control col-xl-4"
                                          id="second_story"
                                          onChange={this.onTextareaChange}
                                          rows="10">
                                </textarea>
                                <textarea className="form-control col-xl-4"
                                          id="third_story"
                                          onChange={this.onTextareaChange}
                                          rows="10">
                                </textarea>
                            </div>

                            <br/>
                            <br/>
                            <div className="row">
                                <h4 className="col-xl-12">Masked Language Model Output With Default BERT</h4>
                                <label className="form-text col-xl-4">First story</label>
                                <label className="form-text col-xl-4">Second story</label>
                                <label className="form-text col-xl-4">Third story</label>

                                {/*<label>Large textarea</label>*/}
                                {/*<label>Large textarea</label>*/}
                            </div>


                            <div className="row" >
                                <textarea className="form-control col-xl-4"
                                          id="exampleFormControlTextarea1"
                                          rows="10" value={this.state.basic[0]}>

                                </textarea>
                                <textarea className="form-control col-xl-4"
                                          id="exampleFormControlTextarea1"
                                          rows="10" value={this.state.basic[1]}>

                                </textarea>
                                <textarea className="form-control col-xl-4"
                                          id="exampleFormControlTextarea1"
                                          rows="10" value={this.state.basic[2]}>

                                </textarea>
                            </div>

                            <br/>
                            <br/>
                            <div className="row">
                                <h4 className="col-xl-12">Masked Language Model Output With Trained BERT</h4>
                                <label className="form-text col-xl-4">First story</label>
                                <label className="form-text col-xl-4">Second story</label>
                                <label className="form-text col-xl-4">Third story</label>
                            </div>


                            <div className="row" >
                                <textarea className="form-control col-xl-4"
                                          id="exampleFormControlTextarea1"
                                          rows="10" value={this.state.trained[0]}>

                                </textarea>
                                <textarea className="form-control col-xl-4"
                                          id="exampleFormControlTextarea1"
                                          rows="10" value={this.state.trained[1]}>

                                </textarea>
                                <textarea className="form-control col-xl-4"
                                          id="exampleFormControlTextarea1"
                                          rows="10"
                                value={this.state.trained[2]}>
                                </textarea>
                            </div>

                            <br/>
                            <br/>
                            <div className="row">
                                <h4 className="col-xl-12">Story generated based on First Story with Masked
                                    LM and NextSentence Prediction</h4>
                                <label className="form-text col-xl-12">Generated Story</label>

                                {/*<label>Large textarea</label>*/}
                                {/*<label>Large textarea</label>*/}
                            </div>
                            <div className="row m-4" >
                                <textarea className="form-control col-xl-12"
                                          id="exampleFormControlTextarea1"
                                          rows="5"
                                            value={this.state.story}>
                                    {this.state.story}
                                </textarea>
                            </div>

                            <br/>
                            <button type="submit">Generate</button>
                        </form>
                        <br/>
                    </div>

            </div>
        )
    }
}



export default SimpleReactFileUpload
